/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-10-26
 *  @Filename: phase_1_column.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use anyhow::{Result, bail};
use polars::{chunked_array::cast::CastOptions, prelude::*};
use std::sync::{Mutex, OnceLock};
use std::{path::PathBuf, thread::available_parallelism};

use crate::hats;
use crate::tools;
use crate::xmatch;
use crate::xmatch::models::JoinPathModel;
use xmatch::models::{CatalogueModel, XMatchModel, XMatchOptions};

static LOG_PREFIX: OnceLock<String> = OnceLock::new();

// Process one join path on the primary key.
// Here "from" is the catalogue being processed and "to" is the catalogue being joined with,
// which must have been previously cross-matched.
pub(crate) fn xmatch_phase1_join_type_column(
    xmatch_path: &PathBuf,
    xmatch_data: &XMatchModel,
    from_cat_data: &CatalogueModel,
    join_path_model: &JoinPathModel,
    xmatch_options: &XMatchOptions,
) -> Result<()> {
    LOG_PREFIX.get_or_init(|| format!("[{}] - Phase 1 -", from_cat_data.name.to_uppercase()));

    let from_join_column = join_path_model.left_on.clone();
    let to_table = join_path_model.right_catalogue.clone();
    let to_join_column = join_path_model.right_on.clone();

    log::info!(
        "{} Processing join path of type 'column': {} -> {}.{}",
        LOG_PREFIX.get().unwrap(),
        from_join_column,
        to_table,
        to_join_column
    );

    // Confirm that the catalogue to join with has been previously cross-matched.
    let to_cat_data = if let Some(cat) = xmatch_data.get_catalogue(to_table.as_str()) {
        cat
    } else {
        bail!(
            "{} Catalogue {} has not been previously cross-matched.",
            LOG_PREFIX.get().unwrap(),
            to_table
        );
    };

    let from_cat_name = &from_cat_data.name.clone();
    let to_cat_name = &to_cat_data.name.clone();
    let to_primary_key = &to_cat_data.primary_key.clone();

    let phase = from_cat_data.get_phase(1).unwrap();

    // Use half of the CPUs if n_workers is not specified.
    let n_workers = phase
        .n_workers
        .unwrap_or_else(|| available_parallelism().unwrap().get() / 2);

    log::info!(
        "{} Joining {}.{} with {}.{} on primary key.",
        LOG_PREFIX.get().unwrap(),
        from_cat_name,
        from_join_column,
        to_cat_name,
        to_primary_key
    );

    let from_hats_path =
        hats::tools::get_hats_path_root(from_cat_data.hats_path.clone())?.join("object/dataset");

    // Get the list of pixels in the from catalogue.
    let partition_info = hats::tools::PartitionInfo::from_dataset(&from_hats_path, true).unwrap();
    let n_pixels = partition_info.len() as u64;

    // Prepare a progress tracker.
    let progress_tracker = if let Some(progress_file) = &xmatch_options.progress_file {
        Some(xmatch::tools::ProgressTracker::new(
            progress_file.clone(),
            n_pixels as i32,
            true,
        ))
    } else {
        None
    };

    let progress_tracker = Mutex::new(progress_tracker);

    // Parallelise the processing of each pixel.
    let pool = crate::tools::create_pool(Some(n_workers))?;
    pool.install(|| {
        use rayon::iter::ParallelIterator;
        use rayon::prelude::*;

        partition_info
            .into_iter()
            .par_bridge()
            .try_for_each(|(order, pixel)| {
                _xmatch_phase1_join_type_column_pixel(
                    order,
                    pixel,
                    xmatch_path,
                    xmatch_data,
                    from_cat_data,
                    from_join_column.as_str(),
                    to_join_column.as_str(),
                    to_cat_data,
                )
                .unwrap();

                // Update the progress tracker.
                if let Some(tracker) = &mut *progress_tracker.lock().unwrap() {
                    tracker.increment().unwrap();
                }

                Ok::<(), anyhow::Error>(())
            })
            .unwrap();
    });

    if let Some(tracker) = &mut *progress_tracker.lock().unwrap() {
        tracker.terminate().unwrap();
    }

    Ok(())
}

// Process one pixel of the from catalogue.
fn _xmatch_phase1_join_type_column_pixel(
    from_order: u8,
    from_pixel: u64,
    xmatch_path: &PathBuf,
    xmatch_data: &XMatchModel,
    from_cat_data: &CatalogueModel,
    from_join_column: &str,
    to_join_column: &str,
    to_cat_data: &CatalogueModel,
) -> Result<()> {
    log::trace!(
        "{} Processing pixel {} of order {}.",
        LOG_PREFIX.get().unwrap(),
        from_pixel,
        from_order
    );

    let xmatch_order = xmatch_data.pixelation_order;

    // Get the list of pixels in the xmatch dataset that correspond to the from pixel.
    // Include neighbouring pixels to account for small differences in on-sky position.
    let xmatch_pixels =
        xmatch::tools::get_xmatch_pixels(xmatch_order, from_order, from_pixel, true)?;

    log::trace!(
        "{} ({}, {}): matched {} pixels in xmatch dataset (order {}): {:?}.",
        LOG_PREFIX.get().unwrap(),
        from_order,
        from_pixel,
        xmatch_pixels.len(),
        xmatch_order,
        xmatch_pixels
    );

    // Scan the original data.
    let cat_dataset_path =
        hats::tools::get_hats_path_root(from_cat_data.hats_path.clone())?.join("object/dataset");

    // Scan the original data for the from pixel.
    let from_lf = if let Some(lf) = hats::tools::scan_pixel_from_hats_dataset(
        cat_dataset_path,
        from_order,
        from_pixel,
        false, // Don't need strict because this is a native order-pixel in this HATS dataset.
    )? {
        lf
    } else {
        return Ok(());
    };

    // We only need the join column and the "from" primary key column.
    let from_primary_key = from_cat_data.primary_key.clone();
    let from_join_column = from_join_column.to_string();
    let to_join_column = to_join_column.to_string();
    let to_primary_key = to_cat_data.primary_key.clone();

    // Filter the from_lf dataset. Remove any rows with null values in the join column.
    let from_lf = from_lf
        .select([col(&from_primary_key), col(&from_join_column)])
        .filter(col(&from_join_column).is_not_null());

    // Exit early if there are no rows with non-null values in the join column.
    let from_lf_nrows = tools::lf_row_count(&from_lf)?;
    if from_lf_nrows == 0 {
        log::trace!(
            "{} ({}, {}): no rows with non-null join column, skipping.",
            LOG_PREFIX.get().unwrap(),
            from_order,
            from_pixel,
        );
        return Ok(());
    }

    // Get the catalog_to lazy frame.
    let catalog_to_table = format!("catalog_to_{}", to_cat_data.name);
    let catalog_to_path =
        xmatch::tools::get_xmatch_path_root(xmatch_path.clone())?.join(&catalog_to_table);

    // Convert the xmatch_pixels to a Series of Int64.
    let xmatch_pixels_series: Series = xmatch_pixels
        .into_iter()
        .collect::<Series>()
        .cast_with_options(&DataType::Int64, CastOptions::NonStrict)?;

    // Filter the catalog_to table to only include rows with the valid pixelation order and pixels.
    let catalog_to_lf =
        LazyFrame::scan_parquet(tools::to_pl_path(catalog_to_path)?, Default::default())?
            .filter(col("Norder").eq(lit(xmatch_data.pixelation_order as u32)))
            .filter(col("Npix").is_in(lit(xmatch_pixels_series).implode(false), false))
            .filter(col("best").eq(lit(true)));

    // If the join column is the primary key, that will be present in the catalog_to table.
    // Otherwise, we need to join with the actual catalogue.
    let catalog_to_lf = if to_join_column != to_primary_key {
        let to_cat_dataset_path =
            hats::tools::get_hats_path_root(to_cat_data.hats_path.clone())?.join("object/dataset");

        let mut to_cat_lf =
            LazyFrame::scan_parquet(tools::to_pl_path(to_cat_dataset_path)?, Default::default())?;

        let to_cat_schema = to_cat_lf.collect_schema()?;
        if !to_cat_schema.contains(&to_join_column) {
            bail!(
                "{} ({}, {}): join column {} not found in catalogue {}.",
                LOG_PREFIX.get().unwrap(),
                from_order,
                from_pixel,
                to_join_column,
                to_cat_data.name
            );
        }

        let to_cat_lf = to_cat_lf.select([col(&to_primary_key), col(&to_join_column)]);

        // Join the catalog_to table with the to catalogue to get the join column.
        catalog_to_lf.join(
            to_cat_lf,
            [col(&to_primary_key)],
            [col(&to_primary_key)],
            JoinArgs {
                how: JoinType::Inner,
                ..Default::default()
            },
        )
    } else {
        catalog_to_lf
    };

    // Get the data type of the join column.
    let from_join_dtype = xmatch::models::get_column_dtype(&from_lf, &from_join_column)?;

    // Perform the join.
    let join_lf = from_lf.join(
        catalog_to_lf,
        [col(&from_join_column)],
        [col(&to_join_column).cast(from_join_dtype)],
        JoinArgs {
            how: JoinType::Inner,
            ..Default::default()
        },
    );

    // Create the catalog_to_table frame.
    let added_on = chrono::Utc::now().timestamp_millis();
    let lf = join_lf.with_columns([
        col("catalogid"),
        col(&from_primary_key),
        col("healpix_29"),
        lit(xmatch_data.version.clone())
            .cast(DataType::String)
            .alias("version"),
        lit(Null {}).cast(DataType::Float64).alias("distance"),
        lit(true).alias("best"),
        lit(1_u8).alias("added_by_phase"),
        lit(added_on).alias("added_on"),
    ]);

    // If the resulting lazy frame has no rows, exit early.
    let lf_nrows = tools::lf_row_count(&lf)?;
    if lf_nrows == 0 {
        log::trace!(
            "{} ({}, {}): no rows after join, skipping.",
            LOG_PREFIX.get().unwrap(),
            from_order,
            from_pixel,
        );
        return Ok(());
    }

    log::trace!(
        "{} ({}, {}): {} rows successful joined.",
        LOG_PREFIX.get().unwrap(),
        from_order,
        from_pixel,
        lf_nrows
    );

    // Write the joined data to the catalog_to_table.
    xmatch::tools::write_catalog_to_table(
        from_cat_data.name.clone(),
        xmatch_path,
        lf,
        from_primary_key.clone(),
        xmatch_order,
        xmatch_data.version.clone(),
        from_cat_data.run_id,
        true,
    )?;

    Ok(())
}
