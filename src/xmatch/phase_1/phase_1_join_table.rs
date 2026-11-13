/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-10-26
 *  @Filename: phase_1_join_table.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use anyhow::Result;
use polars::prelude::*;
use std::sync::{Mutex, OnceLock};
use std::thread::available_parallelism;

use crate::hats::dataset::{AsDataset, Dataset};
use crate::tools::to_pl_path;
use crate::xmatch;
use crate::xmatch::models::{JoinDirection, JoinPathModel, JoinPathTableType, XMatchData};
use crate::xmatch::tools::get_xmatch_path;

use super::phase_1_column::remove_already_xmatched;

static LOG_PREFIX: OnceLock<String> = OnceLock::new();

// Process one join path on the primary key.
// Here "left" is the catalogue that has already been cross-matched and "right" is the
// new catalogue being processed.
pub(crate) fn xmatch_phase1_join_type_table(
    xmatch_data: &XMatchData,
    join_path_model: &JoinPathModel,
) -> Result<()> {
    LOG_PREFIX.get_or_init(|| {
        format!(
            "[{}] - Phase 1 -",
            xmatch_data.catalogue_name.to_uppercase()
        )
    });

    // Collect join table information. We just unwrap here since we have verified that the
    // fields are not none.
    let join_table_path = join_path_model.join_table_path.clone().unwrap();
    let join_table_left_on = join_path_model.join_table_left_on.clone().unwrap();
    let join_table_right_on = join_path_model.join_table_right_on.clone().unwrap();
    let join_table_type = join_path_model.join_table_type.clone();

    // Log the join path information.
    log::info!(
        "{} Processing join path of type 'join_table' with join table in '{}'.",
        LOG_PREFIX.get().unwrap(),
        join_table_path.clone().to_str().unwrap()
    );
    log::debug!(
        "{} Left table: {}.",
        LOG_PREFIX.get().unwrap(),
        join_path_model.left_catalogue
    );
    log::debug!(
        "{} Left join column: {}.",
        LOG_PREFIX.get().unwrap(),
        join_path_model.left_on
    );
    log::debug!(
        "{} Right join column: {}.",
        LOG_PREFIX.get().unwrap(),
        join_path_model.right_on
    );
    log::debug!(
        "{} Join table left_on: {}.",
        LOG_PREFIX.get().unwrap(),
        join_table_left_on
    );
    log::debug!(
        "{} Join table right_on: {}.",
        LOG_PREFIX.get().unwrap(),
        join_table_right_on
    );
    if join_table_type.is_some() {
        log::debug!(
            "{} Join table type: {:?}.",
            LOG_PREFIX.get().unwrap(),
            join_table_type
        );
    }

    // Get the list of pixels in the from catalogue.
    let xmatch_dataset = Dataset::new(
        get_xmatch_path(xmatch_data.path.clone(), None).unwrap(),
        false,
    );
    let xmatch_partition_info = xmatch_dataset.partition();
    let n_pixels = xmatch_partition_info.len() as u64;

    // Prepare a progress tracker.
    let progress_tracker = if let Some(progress_file) = &xmatch_data.options.progress_file {
        Some(xmatch::tools::ProgressTracker::new(
            progress_file.clone(),
            n_pixels as i32,
            true,
        ))
    } else {
        None
    };

    let progress_tracker = Mutex::new(progress_tracker);

    // Parallelise the processing of each pixel in the xmatch prtition.
    let right_cat_model = xmatch_data
        .get_catalogue_model(xmatch_data.catalogue_name.as_str())
        .unwrap();

    let phase = right_cat_model.get_phase(1).unwrap();

    let pool = crate::tools::create_pool(Some(
        phase
            .n_workers
            .unwrap_or_else(|| available_parallelism().unwrap().get() / 2),
    ))?;

    pool.install(|| {
        use rayon::iter::ParallelIterator;
        use rayon::prelude::*;

        xmatch_partition_info
            .into_iter()
            .par_bridge()
            .try_for_each(|(xmatch_order, xmatch_pixel)| {
                xmatch_phase1_join_type_table_pixel(
                    xmatch_order,
                    xmatch_pixel,
                    xmatch_data,
                    join_path_model,
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
fn xmatch_phase1_join_type_table_pixel(
    xmatch_order: u8,
    xmatch_pixel: u64,
    xmatch_data: &XMatchData,
    join_path_model: &JoinPathModel,
) -> Result<()> {
    log::trace!(
        "{} Processing pixel {} (order {}).",
        LOG_PREFIX.get().unwrap(),
        xmatch_pixel,
        xmatch_order
    );

    // Eventually we want to implement joining right to left, but for now we only support
    // joining left to right.
    if join_path_model.join_direction == JoinDirection::RightToLeft {
        anyhow::bail!("Right-to-left joins are not yet supported.");
    }

    // We need to generate the appropriate join depending on the type of join table and whether
    // the join column is the primary key in the catalog_to_<left> table.
    //
    // We start by assuming the join column is the primary key in the catalog_to_<left> table.
    let catalog_to_dataset = Dataset::new(
        get_xmatch_path(
            xmatch_data.path.clone(),
            Some(join_path_model.left_catalogue.clone()),
        )?,
        false,
    );
    let catalog_to_lf = catalog_to_dataset.scan_pixel(xmatch_order, xmatch_pixel, false)?;

    // Read the join table. We assume that the join table may not be in HATS format, so we scan
    // all the parquet files.
    let join_table_path = join_path_model.join_table_path.clone().unwrap();
    let join_table_lf = LazyFrame::scan_parquet(
        to_pl_path(&join_table_path.join("**/*.parquet"))?,
        Default::default(),
    )?;

    // Read the right catalogue. We select all the neighbouring pixels to make sure we get
    // all the rows that may match the input pixel.
    let right_cat_model = xmatch_data
        .get_catalogue_model(xmatch_data.catalogue_name.as_str())
        .unwrap();
    let right_cat_dataset = Dataset::new(right_cat_model.hats_path.clone(), true);
    let right_cat_pixels = right_cat_dataset.get_neighbours(xmatch_order, xmatch_pixel, true)?;
    let right_cat_lf = right_cat_dataset.scan_pixels(right_cat_pixels)?;

    // Let's do all the joins now.
    let joined_lf = catalog_to_lf
        .join(
            join_table_lf,
            [col(join_path_model.left_on.clone())],
            [col(join_path_model.join_table_left_on.clone().unwrap())],
            JoinArgs {
                how: JoinType::Inner,
                ..Default::default()
            },
        )
        .join(
            right_cat_lf,
            [col(join_path_model.join_table_right_on.clone().unwrap())],
            [col(join_path_model.right_on.clone())],
            JoinArgs {
                how: JoinType::Inner,
                ..Default::default()
            },
        );

    // If the left join column is not the primary key in the catalog_to_<left> table, we need to
    // handle it differently.
    let left_cat_model = xmatch_data
        .get_catalogue_model(&join_path_model.left_catalogue)
        .unwrap();

    let joined_lf = if join_path_model.left_on != left_cat_model.primary_key {
        let left_cat_dataset = Dataset::new(left_cat_model.hats_path.clone(), true);
        let left_cat_pixels = left_cat_dataset.get_neighbours(xmatch_order, xmatch_pixel, true)?;
        let left_cat_lf = left_cat_dataset.scan_pixels(left_cat_pixels)?;

        left_cat_lf.join(
            joined_lf,
            [col(left_cat_model.primary_key.clone())],
            [col(left_cat_model.primary_key.clone())],
            JoinArgs {
                how: JoinType::Inner,
                ..Default::default()
            },
        )
    } else {
        joined_lf
    };

    // Finally, we remove entries that were already cross-matched using previous join paths.
    let joined_lf = remove_already_xmatched(joined_lf, xmatch_order, xmatch_pixel, xmatch_data)?;

    // Check the number of rows.
    let lf_nrows = crate::tools::lf_row_count(&joined_lf)?;

    if lf_nrows == 0 {
        log::trace!(
            "{} ({}, {}): no rows after removing duplicates, skipping.",
            LOG_PREFIX.get().unwrap(),
            xmatch_order,
            xmatch_pixel,
        );
        return Ok(());
    }

    log::trace!(
        "{} ({}, {}): {} rows successfully joined.",
        LOG_PREFIX.get().unwrap(),
        xmatch_order,
        xmatch_pixel,
        lf_nrows
    );

    // If the join table is of type Gaia neighbourhood, each row may be matched to multiple
    // neighbours. We use the score column to select the best but we keep the all.
    let lf = if join_path_model
        .join_table_type
        .as_ref()
        .is_some_and(|v| *v == JoinPathTableType::GaiaBestNeighbour)
    {
        joined_lf.with_columns([col("score")
            .eq(col("score").max())
            .over([col("score")])?
            .alias("best")])
    } else {
        joined_lf.with_columns([lit(true).alias("best")])
    };

    let added_on = chrono::Utc::now().timestamp_millis();
    let lf = lf.select([
        col("catalogid"),
        col("_healpix_29"),
        col(&right_cat_model.primary_key).alias("pk"),
        lit(xmatch_data.model.version.clone())
            .cast(DataType::String)
            .alias("version"),
        lit(Null {}).cast(DataType::Float64).alias("distance"),
        lit(true).alias("best"),
        lit(1_u8).alias("added_by_phase"),
        lit(added_on).alias("added_on"),
    ]);

    // Write the joined data to the catalog_to_table.
    xmatch::tools::write_catalog_to_table(
        right_cat_model.name.clone(),
        &xmatch_data.path,
        lf,
        right_cat_model.primary_key.clone(),
        xmatch_order,
        xmatch_data.model.version.clone(),
        right_cat_model.run_id,
        true,
    )?;

    Ok(())
}
