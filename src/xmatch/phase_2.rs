/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-06-15
 *  @Filename: phase_2.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use anyhow::{Result, bail};
use num_format::Locale::en;
use num_format::ToFormattedString;
use polars::prelude::*;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
    thread::available_parallelism,
};

use crate::{
    hats,
    knn::match_sky_radius,
    tools,
    xmatch::{
        self,
        models::{CATALOG_TO_TABLE_SCHEMA, CatalogueModel, XMatchModel, XMatchOptions},
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
        partition_info.len().to_formatted_string(&en)
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
    }

    // Print number of rows added to catalog_to_table.
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
            "{} Finished phase 2 for catalogue {}. catalog_to_{} table has {} rows.",
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

        let tmp_path = xmatch::tools::get_temp_dir_path(
            xmatch_path.clone(),
            &cat_data.name,
            &xmatch_data.version,
            cat_data.run_id as usize,
            true,
        )?;

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

        // Actually move the files.
        xmatch::tools::move_temp_files(
            tmp_catalog_to_table_path.clone(),
            catalog_to_table_path.clone(),
            Some(suffix.clone()),
        )?;

        // Remove the temporary directory if it's empty.
        xmatch::tools::delete_dir_if_empty(tmp_path.parent().unwrap())?;
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
    let primary_key = cat_data.primary_key.clone();
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

    let xmatch_dataset_path = xmatch::tools::get_xmatch_path_root(xmatch_path.clone())?;

    // Need to get the catalog_to_table frame and anti-join so that we keep only the rows in the
    // xmatch catalog table that have not been matched in phase 1.
    let catalog_to_table_path =
        xmatch_dataset_path.join(format!("catalog_to_{}/dataset", cat_data.name));

    // If the catalog_to_table directory does not exist (likely because phase 1 was not run or
    // didn't produce any matches), we create an empty catalog_to_table lazyframe with the
    // needed columns.
    let catalog_to_table_lf = if catalog_to_table_path.exists() {
        xmatch::tools::scan_pixels_from_xmatch_dataset(
            xmatch_dataset_path.clone(),
            Some(cat_data.name.clone()),
            xmatch_order,
            &vec![xmatch_pixel],
        )?
    } else {
        let pk_dtype = cat_lf
            .clone()
            .collect_schema()
            .unwrap()
            .get(&primary_key)
            .unwrap()
            .clone();

        let catalog_to_table_df =
            DataFrame::empty_with_arc_schema(Arc::new(Schema::from_iter(vec![
                Field::new("catalogid".into(), DataType::UInt64),
                Field::new(primary_key.clone().into(), pk_dtype),
            ])));

        catalog_to_table_df.lazy()
    };

    let cat_lf = cat_lf.join(
        catalog_to_table_lf.clone().select([col(&primary_key)]),
        [col(&primary_key)],
        [col(&primary_key)],
        JoinArgs {
            how: JoinType::Anti,
            ..Default::default()
        },
    );

    // Reference epoch column
    let cat_lf = if let Some(ref_epoch) = &cat_data.epoch {
        cat_lf.with_column(
            lit((*ref_epoch).clone())
                .cast(DataType::Float32)
                .alias("ref_epoch"),
        )
    } else {
        cat_lf.with_column(lit(Null {}).cast(DataType::Float32).alias("ref_epoch"))
    };

    let cat_lf = xmatch::tools::propagate_coordinates(cat_lf, &cat_data, to_epoch)?;

    // Select the columns needed for the cross-match and remove nulls.
    let cat_lf = cat_lf
        .select([
            col(&primary_key),
            col(&cat_data.ra_column).cast(DataType::Float64),
            col(&cat_data.dec_column).cast(DataType::Float64),
        ])
        .drop_nans(None)
        .drop_nulls(None);

    // Materialise the catalogue frame.
    let cat_df = cat_lf
        .collect_with_engine(Engine::Streaming)?
        .unwrap_single();

    if cat_df.height() == 0 {
        log::trace!(
            "{} No catalogue data found for this pixel. Skipping.",
            log_prefix
        );
        return Ok(());
    }

    let cat_pk_series = if let Some(series) = cat_df.column(&primary_key)?.as_series() {
        series
    } else {
        log::trace!(
            "{} Failed to extract primary key column. This should not happen.",
            log_prefix
        );
        return Ok(());
    };

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
        xmatch_dataset_path.clone(),
        None,
        xmatch_order,
        &vec![xmatch_pixel],
    )?;

    let xmatch_df = xmatch_lf
        .join(
            catalog_to_table_lf.select([col("catalogid")]),
            [col("catalogid")],
            [col("catalogid")],
            JoinArgs {
                how: JoinType::Anti,
                ..Default::default()
            },
        )
        .select([col("catalogid"), col("healpix_29"), col("ra"), col("dec")])
        .drop_nans(None)
        .drop_nulls(None)
        .collect_with_engine(Engine::Streaming)?
        .unwrap_single();

    if xmatch_df.height() == 0 {
        log::trace!(
            "{} No unmatched xmatch data found for this pixel. Skipping.",
            log_prefix
        );
        return Ok(());
    }

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

    let mut matched_cids: Vec<u64> = Vec::new();
    let mut matched_distances: Vec<f64> = Vec::new();
    let mut matched_best: Vec<bool> = Vec::new();
    let mut matched_cat_pk_series =
        Series::new_empty(PlSmallStr::from("pk"), cat_pk_series.dtype());

    for (cid_idx, match_) in matches.iter().enumerate() {
        if match_.is_empty() {
            continue;
        }

        let distances = match_
            .iter()
            .map(|(distance_rad, _)| distance_rad.to_degrees())
            .collect::<Vec<f64>>();
        let indices = match_
            .iter()
            .map(|(_, idx)| *idx as u32)
            .collect::<Vec<u32>>();

        let pks = cat_pk_series.take(&UInt32Chunked::from_vec("matched_cat_pk".into(), indices))?;
        matched_cat_pk_series.extend(&pks)?;

        let cid_vec = std::iter::repeat(xmatch_cid[cid_idx])
            .take(match_.len())
            .collect::<Vec<u64>>();
        matched_cids.extend(cid_vec);

        matched_distances.extend(distances);

        let mut matched_best_item = std::iter::repeat(false)
            .take(match_.len())
            .collect::<Vec<bool>>();
        matched_best_item[0] = true;
        matched_best.extend(matched_best_item);
    }

    // If there are no matches, skip the pixel.
    if matched_cids.is_empty() {
        log::trace!("{} No matches found for this pixel. Skipping.", log_prefix);
        return Ok(());
    }

    // Build the catalog_to_table frame for the matched rows.
    let catalog_to_table_lf = DataFrame::new(
        matched_cids.len(),
        vec![
            matched_cat_pk_series.into_column(),
            Series::new("catalogid".into(), matched_cids).into_column(),
            Series::new("distance".into(), matched_distances).into_column(),
            Series::new("best".into(), matched_best).into_column(),
        ],
    )?
    .lazy();

    // Join with the catalog table to get the healpix_29
    let catalog_to_table_lf = catalog_to_table_lf.join(
        xmatch_df
            .lazy()
            .select([col("catalogid"), col("healpix_29")]),
        [col("catalogid")],
        [col("catalogid")],
        JoinArgs {
            how: JoinType::Left,
            ..Default::default()
        },
    );

    // Add other columns.
    let added_on = xmatch::tools::get_unix_timestamp_ms();
    let catalog_to_table_lf = catalog_to_table_lf
        .with_column(lit(2).alias("added_by_phase"))
        .with_column(lit(added_on).alias("added_on"))
        .with_column(lit(xmatch_data.version.clone()).alias("version"));

    let catalog_to_table_lf = xmatch::models::cast_and_reorder(
        catalog_to_table_lf,
        &CATALOG_TO_TABLE_SCHEMA,
        Some(primary_key),
    )?;

    // Write the catalog_to_table frame to the xmatch dataset.
    let tmp_path = xmatch::tools::get_temp_dir_path(
        xmatch_path.clone(),
        &cat_data.name,
        &xmatch_data.version,
        cat_data.run_id as usize,
        true,
    )?
    .join(format!("catalog_to_{}", cat_data.name));
    std::fs::create_dir_all(&tmp_path)?;

    xmatch::tools::sink_lazyframe_pixel(
        catalog_to_table_lf,
        tmp_path.clone().join("dataset"),
        xmatch_order,
        xmatch_pixel,
        true,
    )?;

    Ok(())
}
