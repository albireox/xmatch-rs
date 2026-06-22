/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-06-15
 *  @Filename: phase_2.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use anyhow::{Result, bail};
use polars::prelude::*;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
    thread::available_parallelism,
};

use crate::{
    hats,
    knn::match_sky_radius,
    xmatch::{
        self,
        models::{CatalogueModel, XMatchModel, XMatchOptions},
    },
};

static LOG_PREFIX: OnceLock<String> = OnceLock::new();

pub fn run_phase_2(
    xmatch_path: PathBuf,
    xmatch_data: XMatchModel,
    cat_data: CatalogueModel,
    xmatch_options: XMatchOptions,
) -> Result<()> {
    LOG_PREFIX.get_or_init(|| format!("[{}] - Phase 2 -", cat_data.name.to_uppercase()));

    let phase = if let None = cat_data.get_phase(2) {
        bail!(
            "{} Phase 2 is not defined for catalogue model {}",
            LOG_PREFIX.get().unwrap(),
            cat_data.name
        );
    } else {
        cat_data.get_phase(2).unwrap()
    };

    log::info!(
        "{} Running phase 2 for catalogue {}.",
        LOG_PREFIX.get().unwrap(),
        cat_data.name
    );

    // Get the list of pixels in the xmatch dataset. For spatial cross-marching we cross-match the
    // xmatch catalog pixels against the catalogue pixels (catalogid -> target_id).
    let xmatch_dataset =
        xmatch::tools::get_xmatch_path_root(xmatch_path.clone())?.join("catalog/dataset");
    let partition_info = hats::tools::PartitionInfo::from_dataset(&xmatch_dataset, false)?;

    log::debug!(
        "{} Found {} pixels in the xmatch dataset.",
        LOG_PREFIX.get().unwrap(),
        partition_info.len()
    );

    // Prepare a progress tracker.
    let progress_tracker = if let Some(progress_file) = &xmatch_options.progress_file {
        Some(xmatch::tools::ProgressTracker::new(
            progress_file.clone(),
            partition_info.len() as i32,
            true,
        ))
    } else {
        None
    };

    let progress_tracker = Arc::new(Mutex::new(progress_tracker));

    // Use a quarter of the CPUs if n_workers is not specified.
    let n_workers = phase
        .n_workers
        .unwrap_or_else(|| available_parallelism().unwrap().get() / 4);

    // Parallelise the processing of each pixel.
    let pool = crate::tools::create_pool(Some(n_workers))?;
    pool.install(|| {
        use rayon::iter::ParallelIterator;
        use rayon::prelude::*;

        partition_info
            .into_iter()
            .par_bridge()
            .try_for_each(|(order, pixel)| {
                _xmatch_phase2_pixel(
                    order,
                    pixel,
                    &xmatch_path,
                    &xmatch_data,
                    &cat_data,
                    phase.query_radius.unwrap_or(1.0f64),
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
        // Sleep a bit to allow the Python process to remove the progress bar.
        std::thread::sleep(std::time::Duration::from_secs(3));
    }

    Ok(())
}

fn _xmatch_phase2_pixel(
    xmatch_order: u8,
    xmatch_pixel: u64,
    xmatch_path: &PathBuf,
    xmatch_data: &XMatchModel,
    cat_data: &CatalogueModel,
    query_radius: f64,
) -> Result<()> {
    let log_prefix = format!(
        "{} ({}, {}):",
        LOG_PREFIX.get().unwrap(),
        xmatch_order,
        xmatch_pixel
    );

    log::trace!(
        "{} Processing xmatch pixel {} of order {} for catalogue {}.",
        log_prefix,
        xmatch_pixel,
        xmatch_order,
        cat_data.name
    );

    // Get the catalogue lazyframe for the pixel to process.
    let hats_dataset_path =
        xmatch::tools::get_hats_path_root(cat_data.hats_path.clone())?.join("object/dataset");
    let cat_lf = xmatch::tools::scan_pixel_from_hats_dataset(
        hats_dataset_path,
        xmatch_order,
        xmatch_pixel,
        false,
    )?;

    let cat_lf = if let Some(cat_lf) = cat_lf {
        cat_lf
    } else {
        log::trace!(
            "{} No catalogue data found for pixel {} of order {}. Skipping.",
            log_prefix,
            xmatch_pixel,
            xmatch_order
        );
        return Ok(());
    };

    // Propagate coordinates to the epoch of the xmatch dataset.
    let to_epoch = xmatch_data.epoch;
    let cat_lf = xmatch::tools::propagate_coordinates(cat_lf, &cat_data, to_epoch)?;

    // Select the columns needed for the cross-match and remove nulls.
    let cat_lf = cat_lf
        .select([
            col(&cat_data.primary_key),
            col(&cat_data.ra_column),
            col(&cat_data.dec_column),
        ])
        .drop_nans(None)
        .drop_nulls(None);

    // Materialise the catalogue frame.
    let cat_df = cat_lf
        .collect_with_engine(Engine::Streaming)?
        .unwrap_single();

    let cat_ra = cat_df
        .column(&cat_data.ra_column)?
        .f64()?
        .into_no_null_iter();
    let cat_dec = cat_df
        .column(&cat_data.dec_column)?
        .f64()?
        .into_no_null_iter();

    let cat_coords = cat_ra.zip(cat_dec).collect::<Vec<(f64, f64)>>();

    // Get the xmatch pixel frame and coordinate vectors.
    let xmatch_dataset_path = xmatch::tools::get_xmatch_path_root(xmatch_path.clone())?;

    let xmatch_lf = xmatch::tools::scan_pixels_from_xmatch_dataset(
        xmatch_dataset_path.clone().join("catalog/dataset"),
        None,
        xmatch_order,
        &vec![xmatch_pixel],
    )?;

    // Need to get the catalog_to_table frame and anti-join so that we keep only the rows in the
    // xmatch catalog table that have not been matched in phase 1.
    // let catalog_to_table_lf = xmatch::tools::get_catalog_to_table_lf(
    //     xmatch_dataset_path.clone().join("catalog/catalog_to_table"),
    //     xmatch_order,
    //     xmatch_pixel,
    // )?;

    let xmatch_df = xmatch_lf
        .select([col("catalogid"), col("ra"), col("dec")])
        .drop_nans(None)
        .drop_nulls(None)
        .collect_with_engine(Engine::Streaming)?
        .unwrap_single();

    let xmatch_cid = xmatch_df
        .column("catalogid")?
        .u64()?
        .into_no_null_iter()
        .collect::<Vec<u64>>();

    let xmatch_ra = xmatch_df.column("ra")?.f64()?.into_no_null_iter();
    let xmatch_dec = xmatch_df.column("dec")?.f64()?.into_no_null_iter();

    let xmatch_coords = xmatch_ra.zip(xmatch_dec).collect::<Vec<(f64, f64)>>();

    // Convert coordinates to radians.
    let cat_coords_rad = cat_coords
        .into_iter()
        .map(|(ra, dec)| (ra.to_radians(), dec.to_radians()))
        .collect::<Vec<(f64, f64)>>();

    let xmatch_coords_rad = xmatch_coords
        .into_iter()
        .map(|(ra, dec)| (ra.to_radians(), dec.to_radians()))
        .collect::<Vec<(f64, f64)>>();

    // Calculate the cross-match between the catalogue and xmatch pixel coordinates.
    let matches = match_sky_radius(
        &cat_coords_rad,
        &xmatch_coords_rad,
        (query_radius / 3600.0f64).to_radians(),
    )?;

    Ok(())
}
