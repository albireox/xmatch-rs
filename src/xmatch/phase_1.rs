/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-06-15
 *  @Filename: phase_1.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use anyhow::{Result, bail};
use core::time;
use num_format::Locale::en;
use num_format::ToFormattedString;
use polars::{chunked_array::cast::CastOptions, prelude::*};
use regex::Regex;
use std::sync::{Mutex, OnceLock};
use std::thread::sleep;
use std::{collections::HashMap, path::PathBuf, thread::available_parallelism};

use crate::hats;
use crate::tools;
use crate::xmatch;
use crate::xmatch::tools::ProgressTracker;
use xmatch::models::{CatalogueModel, XMatchModel, XMatchOptions};

static LOG_PREFIX: OnceLock<String> = OnceLock::new();

pub fn run_phase_1(
    xmatch_path: PathBuf,
    xmatch_data: XMatchModel,
    cat_data: CatalogueModel,
    xmatch_options: XMatchOptions,
) -> Result<()> {
    LOG_PREFIX.get_or_init(|| format!("[{}] - Phase 1 -", cat_data.name.to_uppercase()));

    let phase = if let None = cat_data.get_phase(1) {
        panic!(
            "{} Phase 1 is not defined for catalogue model {}",
            LOG_PREFIX.get().unwrap(),
            cat_data.name
        );
    } else {
        cat_data.get_phase(1).unwrap()
    };

    log::info!(
        "{} Running phase 1 for catalogue model {}",
        LOG_PREFIX.get().unwrap(),
        cat_data.name
    );

    // Check join paths.
    let join_paths = &phase.join_paths;
    if join_paths.is_empty() {
        log::info!(
            "{} No join paths defined for phase 1, skipping phase.",
            LOG_PREFIX.get().unwrap()
        );
        return Ok(());
    }

    let mut join_path_map = HashMap::<String, (String, String)>::new();

    let re_from_path = Regex::new(r"^([a-zA-Z0-9_]+)$").unwrap();
    let re_to_path = Regex::new(r"^([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)$").unwrap();

    // Reformat join paths to a hash map of from_column -> (to_table, to_column).
    for join_path in join_paths {
        let from_join_column = join_path.0.clone();
        let to_table_column = join_path.1.clone();

        if let None = re_from_path.captures(&from_join_column) {
            bail!(
                "{} Invalid from column format for phase 1: {}.'",
                LOG_PREFIX.get().unwrap(),
                from_join_column
            );
        }

        if let Some(caps) = re_to_path.captures(&to_table_column) {
            let to_table = caps.get(1).unwrap().as_str().to_string();
            let to_join_column = caps.get(2).unwrap().as_str().to_string();
            join_path_map.insert(from_join_column, (to_table, to_join_column));
        } else {
            bail!(
                "{} Invalid join path format for phase 1: {}.'",
                LOG_PREFIX.get().unwrap(),
                to_table_column
            );
        }
    }

    // Process each join path.
    for (from_join_column, (to_table, to_join_column)) in join_path_map {
        _xmatch_phase1_process_join_path(
            &xmatch_path,
            &xmatch_data,
            &cat_data,
            &from_join_column,
            &to_table,
            &to_join_column,
            &xmatch_options,
        )?;
    }

    // Report the number of rows in the catalog_to_table after processing all join paths.
    let tmp_path = xmatch::tools::get_temp_dir_path(
        xmatch_path.clone(),
        &cat_data.name,
        &xmatch_data.version.clone(),
        cat_data.run_id as usize,
        true,
    )?;

    let catalog_to_table_name = format!("catalog_to_{}", cat_data.name);

    let catalog_to_table_path = xmatch::tools::get_xmatch_path_root(xmatch_path.clone())?
        .join(format!("{}/dataset", catalog_to_table_name));

    if catalog_to_table_path.exists() {
        let catalog_to_lf = LazyFrame::scan_parquet(
            tools::to_pl_path(catalog_to_table_path)?,
            Default::default(),
        )?;

        let n_rows = tools::lf_row_count(&catalog_to_lf)?;

        log::info!(
            "{} Finished phase 1 for catalogue {}. catalog_to_{} table has {} rows.",
            LOG_PREFIX.get().unwrap(),
            cat_data.name,
            cat_data.name,
            n_rows.to_formatted_string(&en)
        );
    }

    // Move the original catalogue files if requested.
    if xmatch_options.move_files {
        log::info!(
            "{} Moving temporary catalog_to_{} files to xmatch dataset.",
            LOG_PREFIX.get().unwrap(),
            cat_data.name
        );

        let suffix = format!(
            "{}-{}-{}",
            cat_data.name, xmatch_data.version, cat_data.run_id
        );

        // catalog_to_table directory
        let catalog_to_table_name = format!("catalog_to_{}", cat_data.name);

        let catalog_to_table_path = xmatch::tools::get_xmatch_path_root(xmatch_path.clone())?
            .join(format!("{}/dataset", catalog_to_table_name));
        std::fs::create_dir_all(&catalog_to_table_path)?;

        let tmp_catalog_to_table_path = tmp_path.join(format!("{}/dataset", catalog_to_table_name));

        log::debug!(
            "{} Moving {} dataset from {} to {}.",
            LOG_PREFIX.get().unwrap(),
            catalog_to_table_name,
            tmp_catalog_to_table_path.display(),
            catalog_to_table_path.display()
        );

        xmatch::tools::move_temp_files(
            tmp_catalog_to_table_path.clone(),
            catalog_to_table_path.clone(),
            Some(suffix.clone()),
        )?;
    }
    Ok(())
}

// Process one join path.
fn _xmatch_phase1_process_join_path(
    xmatch_path: &PathBuf,
    xmatch_data: &XMatchModel,
    from_cat_data: &CatalogueModel,
    from_join_column: &str,
    to_table: &str,
    to_join_column: &str,
    xmatch_options: &XMatchOptions,
) -> Result<()> {
    log::info!(
        "{} Processing join path {} -> {}.{}",
        LOG_PREFIX.get().unwrap(),
        from_join_column,
        to_table,
        to_join_column
    );

    // Confirm that the catalogue to join with has been previously cross-matched.
    let to_cat_data = if let Some(cat) = xmatch_data.get_catalogue(to_table) {
        cat
    } else {
        bail!(
            "{} Catalogue {} has not been previously cross-matched.",
            LOG_PREFIX.get().unwrap(),
            to_table
        );
    };

    // For now we only support joining on primary keys. This does not require access to
    // the original catalogue data because all the information is present in the
    // catalog_to_<to_catalogue> table.
    if to_cat_data.primary_key == to_join_column {
        log::debug!(
            "{} Join column {} is the primary key of catalogue {}.",
            LOG_PREFIX.get().unwrap(),
            to_join_column,
            to_table
        );
        _xmatch_phase1_join_pk(
            xmatch_path,
            xmatch_data,
            from_cat_data,
            to_cat_data,
            from_join_column,
            xmatch_options,
        )?;
    } else {
        bail!(
            "{} Non-primary key join not yet implemented: {}.{} -> {}.{}",
            LOG_PREFIX.get().unwrap(),
            from_cat_data.name,
            from_join_column,
            to_table,
            to_join_column
        );
    }

    Ok(())
}

// Process one join path on the primary key.
fn _xmatch_phase1_join_pk(
    xmatch_path: &PathBuf,
    xmatch_data: &XMatchModel,
    from_cat_data: &CatalogueModel,
    to_cat_data: &CatalogueModel,
    from_join_column: &str,
    xmatch_options: &XMatchOptions,
) -> Result<()> {
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
        xmatch::tools::get_hats_path_root(from_cat_data.hats_path.clone())?.join("object/dataset");

    // Get the list of pixels in the from catalogue.
    let partition_info = hats::tools::PartitionInfo::from_dataset(&from_hats_path, true).unwrap();
    let n_pixels = partition_info.len() as u64;

    // Prepare a progress tracker.
    let progress_tracker = if let Some(progress_file) = &xmatch_options.progress_file {
        Some(xmatch::tools::ProgressTracker::new(
            progress_file.clone(),
            n_pixels as i32,
        ))
    } else {
        None
    };

    let progress_tracker = Arc::new(Mutex::new(progress_tracker));

    // Parallelise the processing of each pixel.
    let pool = crate::tools::create_pool(Some(n_workers))?;
    pool.install(|| {
        use rayon::iter::ParallelIterator;
        use rayon::prelude::*;

        partition_info
            .into_iter()
            .par_bridge()
            .try_for_each(|(order, pixel)| {
                _xmatch_phase1_join_pk_pixel(
                    order,
                    pixel,
                    xmatch_path,
                    xmatch_data,
                    from_cat_data,
                    from_join_column,
                    to_cat_data,
                    progress_tracker.clone(),
                )
            })
            .unwrap();
    });

    if let Some(tracker) = &mut *progress_tracker.lock().unwrap() {
        tracker.terminate().unwrap();
        // Sleep a bit to allow the Python process to remove the progress bar.
        sleep(time::Duration::from_secs(3));
    }

    Ok(())
}

// Process one pixel of the from catalogue.
fn _xmatch_phase1_join_pk_pixel(
    from_order: u8,
    from_pixel: u64,
    xmatch_path: &PathBuf,
    xmatch_data: &XMatchModel,
    from_cat_data: &CatalogueModel,
    from_join_column: &str,
    to_cat_data: &CatalogueModel,
    progress_tracker: Arc<Mutex<Option<xmatch::tools::ProgressTracker>>>,
) -> Result<()> {
    let increment_progress = || {
        if let Some(tracker) = &mut *progress_tracker.lock().unwrap() {
            tracker.increment().unwrap();
        }
    };

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
        xmatch::tools::get_hats_path_root(from_cat_data.hats_path.clone())?.join("object/dataset");

    // Scan the original data for the from pixel.
    let from_lf = if let Some(lf) = xmatch::tools::scan_pixel_from_hats_dataset(
        cat_dataset_path,
        from_order,
        from_pixel,
        false, // Don't need strict because this is a native order-pixel in this HATS dataset.
    )? {
        lf
    } else {
        increment_progress();
        return Ok(());
    };

    // We only need the join column and the "from" primary key column.
    let from_primary_key = from_cat_data.primary_key.clone();
    let from_join_column = from_join_column.to_string();
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
        increment_progress();
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
            .filter(col("Npix").is_in(lit(xmatch_pixels_series).implode(false), false));

    // Get the data type of the join column.
    let from_join_dtype = xmatch::models::get_column_dtype(&from_lf, &from_join_column)?;

    // Perform the join.
    let join_lf = from_lf.join(
        catalog_to_lf,
        [col(&from_join_column)],
        [col(&to_primary_key).cast(from_join_dtype)],
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
        lit(added_on)
            .cast(DataType::Datetime(TimeUnit::Milliseconds, None))
            .alias("added_on"),
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
        increment_progress();
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

    increment_progress();
    Ok(())
}
