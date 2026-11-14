/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-06-15
 *  @Filename: phase_2.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use std::sync::{Arc, Mutex};
use std::thread::available_parallelism;

use anyhow::{Result, bail};
use num_format::Locale::en;
use num_format::ToFormattedString;
use polars::prelude::*;

use crate::hats::dataset::{AsDataset, Dataset};
use crate::knn::match_sky_radius;
use crate::tools::{self, IsEmpty};
use crate::xmatch::models::{
    CATALOG_TO_TABLE_SCHEMA, CatalogueModel, CoordUnit, EpochFormat, XMatchData, cast_and_reorder,
};
use crate::xmatch::tools as xmatch_tools;

pub(crate) fn run_phase_2(xmatch_data: XMatchData) -> Result<()> {
    let log_prefix = format!(
        "[{}] - Phase 2 -",
        xmatch_data.catalogue_name.to_uppercase()
    );

    let cat_model = xmatch_data
        .get_catalogue_model(&xmatch_data.catalogue_name.as_str())
        .unwrap();

    let phase = if let None = cat_model.get_phase(2) {
        bail!(
            "{} Phase 2 is not defined for catalogue model {}",
            log_prefix,
            cat_model.name
        );
    } else {
        cat_model.get_phase(2).unwrap()
    };

    log::info!(
        "{} Running phase 2 for catalogue {}.",
        log_prefix,
        cat_model.name
    );

    // Get the list of pixels in the new catalogue. In phase 2 we cross-match right-to-left
    // (new catalogue to already cross-matched catalogids) because we expect catalogues to be
    // added in order of increasing spatial resolution.
    let cat_dataset = Dataset::new(cat_model.hats_path.clone(), true);
    let partition_info = cat_dataset.partition();

    log::debug!(
        "{} Found {} pixels in the catalogue dataset.",
        log_prefix,
        partition_info.len().to_formatted_string(&en)
    );

    // Prepare a progress tracker.
    let progress_tracker = if let Some(progress_file) = &xmatch_data.options.progress_file {
        Some(crate::xmatch::tools::ProgressTracker::new(
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
                    &xmatch_data,
                    phase.query_radius.unwrap_or(1.0f64) as f32,
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
    let catalog_to_table_path =
        xmatch_tools::get_xmatch_path(xmatch_data.path.clone(), Some(cat_model.name.clone()))?;

    if catalog_to_table_path.exists() {
        let catalog_to_lf = LazyFrame::scan_parquet(
            tools::to_pl_path(catalog_to_table_path)?,
            Default::default(),
        )?;

        let n_rows = tools::lf_row_count(&catalog_to_lf)?;

        log::info!(
            "{} Finished phase 2 for catalogue {}. catalog_to_{} table has {} rows.",
            log_prefix,
            cat_model.name,
            cat_model.name,
            n_rows.to_formatted_string(&en)
        );
    }

    // Move the original catalogue files if requested.
    if xmatch_data.options.move_files {
        log::info!(
            "{} Moving temporary catalog_to_{} files to xmatch dataset.",
            log_prefix,
            cat_model.name
        );

        let tmp_path = xmatch_tools::get_temp_dir_path(
            xmatch_data.path.clone(),
            &cat_model.name,
            &xmatch_data.model.version,
            cat_model.run_id as usize,
            true,
        )?;

        let suffix = format!(
            "{}-{}-{}",
            cat_model.name, xmatch_data.model.version, cat_model.run_id
        );

        // catalog_to_table directory
        let catalog_to_table_name = format!("catalog_to_{}", cat_model.name);

        let catalog_to_table_path =
            xmatch_tools::get_xmatch_path(xmatch_data.path.clone(), Some(cat_model.name.clone()))?;
        std::fs::create_dir_all(&catalog_to_table_path)?;

        let tmp_catalog_to_table_path = tmp_path.join(format!("{}/dataset", catalog_to_table_name));

        log::debug!(
            "{} Moving {} dataset from {} to {}.",
            log_prefix,
            catalog_to_table_name,
            tmp_catalog_to_table_path.display(),
            catalog_to_table_path.display()
        );

        // Actually move the files.
        xmatch_tools::move_temp_files(
            tmp_catalog_to_table_path.clone(),
            catalog_to_table_path.clone(),
            Some(suffix.clone()),
        )?;

        // Remove the temporary directory if it's empty.
        xmatch_tools::delete_dir_if_empty(tmp_path.parent().unwrap())?;
    }

    Ok(())
}

fn _xmatch_phase2_pixel(
    order: u8,
    pixel: u64,
    xmatch_data: &XMatchData,
    query_radius: f32,
) -> Result<()> {
    let log_prefix = format!(
        "[{}] - Phase 1 -",
        xmatch_data.catalogue_name.to_uppercase()
    );
    let log_prefix = format!("{} ({}, {}):", log_prefix, order, pixel);

    log::trace!(
        "{} Processing pixel {} of order {} for catalogue {}.",
        log_prefix,
        pixel,
        order,
        xmatch_data.catalogue_name
    );

    let cat_model = xmatch_data
        .get_catalogue_model(xmatch_data.catalogue_name.as_str())
        .unwrap();

    // Get the catalogue lazyframe for the pixel to process.
    let primary_key = cat_model.primary_key.clone();

    let hats_dataset = Dataset::new(cat_model.hats_path.clone(), true);
    let cat_lf = hats_dataset.scan_pixel(order, pixel, false)?;

    if cat_lf.is_empty() {
        log::trace!(
            "{} No catalogue data found for pixel {} of order {}. Skipping.",
            log_prefix,
            pixel,
            order
        );
        return Ok(());
    };

    // Need to get the catalog_to_table frame and anti-join so that we keep only the rows in the
    // xmatch catalog table that have not been matched in phase 1.
    let catalog_to_table_path =
        xmatch_tools::get_xmatch_path(xmatch_data.path.clone(), Some(cat_model.name.clone()))?;

    // If the catalog_to_table directory does not exist (likely because phase 1 was not run or
    // didn't produce any matches), we create an empty catalog_to_table lazyframe with the
    // needed columns.
    let pk_dtype = cat_lf
        .clone()
        .collect_schema()
        .unwrap()
        .get(&primary_key)
        .unwrap()
        .clone();

    let catalog_to_table_placeholder_df =
        DataFrame::empty_with_arc_schema(Arc::new(Schema::from_iter(vec![
            Field::new("catalogid".into(), DataType::UInt64),
            Field::new(primary_key.clone().into(), pk_dtype),
        ])));

    let catalog_to_table_lf = if catalog_to_table_path.exists() {
        let catalog_to_table_dataset = Dataset::new(catalog_to_table_path, false);

        let catalog_to_table_pixels =
            catalog_to_table_dataset.get_neighbours(order, pixel, true)?;

        if catalog_to_table_pixels.is_empty() {
            catalog_to_table_placeholder_df.clone().lazy()
        } else {
            let catalog_to_table_lf =
                catalog_to_table_dataset.scan_pixels(catalog_to_table_pixels)?;
            if catalog_to_table_lf.is_empty() {
                catalog_to_table_placeholder_df.clone().lazy()
            } else {
                catalog_to_table_lf
            }
        }
    } else {
        catalog_to_table_placeholder_df.clone().lazy()
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

    // We need to estimate the maximum proper motion offset for the targets being cross-matched.
    let to_epoch = xmatch_data.model.epoch;
    let max_pm_offset = get_max_pm_offset(&cat_lf, &cat_model, to_epoch)?;

    // Propagate coordinates to the epoch of the xmatch dataset.
    let cat_lf = xmatch_tools::propagate_coordinates(cat_lf, &cat_model, to_epoch)?;

    // Select the columns needed for the cross-match and remove nulls.
    let cat_lf = cat_lf
        .select([
            col(&primary_key),
            col("ra_epoch").cast(DataType::Float64),
            col("dec_epoch").cast(DataType::Float64),
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

    let cat_ra = cat_df.column("ra_epoch")?.f64()?.into_no_null_iter();
    let cat_dec = cat_df.column("dec_epoch")?.f64()?.into_no_null_iter();

    let cat_coords = cat_ra.zip(cat_dec).collect::<Vec<(f64, f64)>>();

    // Define the buffer around the xmatch pixels as the query radius (times two just in case)
    // plus the maximum proper motion offset.
    let buffer = (query_radius * 2.0) + max_pm_offset;

    // Get the associated xmatch frame and coordinate vectors.
    let xmatch_dataset = Dataset::new(
        xmatch_tools::get_xmatch_path(xmatch_data.path.clone(), None)?,
        false,
    );

    // Select the xmatch data within the buffer.
    let xmatch_lf = xmatch_dataset.get_pixel_with_buffer(order, pixel, buffer)?;

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
        .select([col("catalogid"), col("_healpix_29"), col("ra"), col("dec")])
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
        &xmatch_coords_rad,
        &cat_coords_rad,
        (query_radius / 3600.0f32).to_radians() as f64,
    )?;

    let mut matched_cids: Vec<u64> = Vec::new();
    let mut matched_distances: Vec<f64> = Vec::new();
    let mut matched_best: Vec<bool> = Vec::new();
    let mut matched_cat_pk_series =
        Series::new_empty(PlSmallStr::from("pk"), cat_pk_series.dtype());

    // Loop over each match. "matches" has the size of the catalogue coordinate array
    // and each match is a vector of (distance, catalogid_index).
    for (cat_idx, match_) in matches.iter().enumerate() {
        if match_.is_empty() {
            continue;
        }

        // Get the primary key for the catalogue row. We do this using series instead of vectors
        // because we don't know the primary key data type.
        let cat_pk = cat_pk_series.take(&UInt32Chunked::from_vec(
            "matched_cat_pk".into(),
            vec![cat_idx as u32].repeat(match_.len()),
        ))?;
        matched_cat_pk_series.extend(&cat_pk)?;

        let (distances, indices): (Vec<f64>, Vec<usize>) = match_
            .iter()
            .map(|(distance_rad, idx)| (distance_rad.to_degrees(), *idx as usize))
            .unzip();

        let cid_vec = indices
            .iter()
            .map(|&idx| xmatch_cid[idx])
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
            .select([col("catalogid"), col("_healpix_29")]),
        [col("catalogid")],
        [col("catalogid")],
        JoinArgs {
            how: JoinType::Left,
            ..Default::default()
        },
    );

    // Add other columns.
    let added_on = xmatch_tools::get_unix_timestamp_ms();
    let catalog_to_table_lf = catalog_to_table_lf
        .with_column(lit(2).alias("added_by_phase"))
        .with_column(lit(added_on).alias("added_on"))
        .with_column(lit(xmatch_data.model.version.clone()).alias("version"));

    let catalog_to_table_lf = cast_and_reorder(
        catalog_to_table_lf,
        &CATALOG_TO_TABLE_SCHEMA,
        Some(primary_key),
    )?;

    // Write the catalog_to_table frame to the xmatch dataset.
    let tmp_path = xmatch_tools::get_temp_dir_path(
        xmatch_data.path.clone(),
        &cat_model.name.clone(),
        &xmatch_data.model.version,
        cat_model.run_id as usize,
        true,
    )?
    .join(format!("catalog_to_{}", cat_model.name));
    std::fs::create_dir_all(&tmp_path)?;

    // The catalog_to_table frame corresponds to the new catalogue (order, pixel) but catalog_to
    // datasets are partitioned by xmatch order.
    let factor = 4u64.pow((29 - xmatch_data.model.pixelation_order) as u32);
    let catalog_to_table_lf_xm = catalog_to_table_lf.with_column(
        col("_healpix_29")
            .true_div(lit(factor))
            .cast(DataType::UInt64)
            .alias("xmatch_pixel"),
    );

    let xmatch_pixel_df = catalog_to_table_lf_xm
        .clone()
        .select([col("xmatch_pixel")])
        .collect()?;

    let mut xmatch_pixels = xmatch_pixel_df
        .column("xmatch_pixel")?
        .u64()?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    xmatch_pixels.sort_unstable();
    xmatch_pixels.dedup();

    for xmatch_pixel in xmatch_pixels {
        let pixel_lf = catalog_to_table_lf_xm
            .clone()
            .filter(col("xmatch_pixel").eq(lit(xmatch_pixel)))
            .drop(Selector::ByName {
                names: Arc::new(["xmatch_pixel".into()]),
                strict: true,
            });

        xmatch_tools::sink_lazyframe_pixel(
            pixel_lf,
            tmp_path.clone().join("dataset"),
            xmatch_data.model.pixelation_order,
            xmatch_pixel,
            true,
        )?;
    }

    Ok(())
}

// Returns the maximum offset due to proper motion in arcsec.
fn get_max_pm_offset(cat_lf: &LazyFrame, cat_model: &CatalogueModel, to_epoch: f32) -> Result<f32> {
    let lf = cat_lf.clone();

    let pmra_col = cat_model.pmra_column.clone();
    let pmdec_col = cat_model.pmdec_column.clone();

    if pmra_col.is_none() || pmdec_col.is_none() {
        return Ok(0.0);
    }

    let pmra_col = pmra_col.unwrap();
    let pmdec_col = pmdec_col.unwrap();

    // If pmra is not a great circle angle, convert it now.
    let lf = if cat_model.pmra_is_cosdec.is_some() && !cat_model.pmra_is_cosdec.unwrap() {
        // Ensure dec is in degrees.
        let dec_col = cat_model.dec_column.clone();

        let lf = match cat_model.dec_column_unit {
            CoordUnit::Radians => lf.with_column(col(&dec_col).degrees().alias("dec")),
            _ => lf.with_column(col(&dec_col).alias("dec")),
        };

        // cos_dec factor
        let cos_dec = col("dec").radians().cos();

        lf.with_column(col(&pmra_col) * cos_dec.clone())
    } else {
        lf
    };

    // Epoch difference. We have two cases to consider: either the reference epoch is a single
    // value or a column, and it could be in Julian date or JD.
    let ref_epoch = if let Some(ref_epoch) = &cat_model.epoch {
        lit((*ref_epoch).clone())
    } else if let Some(ref_epoch) = &cat_model.epoch_column {
        col(ref_epoch)
    } else {
        bail!("Reference epoch is not specified");
    };

    // If the ref_epoch values are in JD, convert them to Julian year.
    // 2451545 is the JD for 2000.0 and a Julian year is 365.25 days.
    let ref_epoch = if cat_model.epoch_format.is_some()
        && cat_model.epoch_format.clone().unwrap() == EpochFormat::JD
    {
        (ref_epoch - lit(2451545.0)) / lit(365.25) + lit(2000.0)
    } else {
        ref_epoch
    };

    let epoch_diff = (lit(to_epoch) - ref_epoch).fill_nan(0.0).fill_null(0.0);

    // ra difference in deg
    let ra_diff = col(&pmra_col).fill_null(0.0).fill_null(0.0) * epoch_diff.clone()
        / lit(3600.0)
        / lit(1000.0);

    // dec difference in deg.
    let dec_diff = col(&pmdec_col).fill_null(0.0).fill_null(0.0) * epoch_diff.clone()
        / lit(3600.0)
        / lit(1000.0);

    // Get the maximum pm offset.
    let max_pm_offset = lf
        .with_columns([(ra_diff.pow(2) + dec_diff.pow(2)).sqrt().alias("pm_offset")])
        .select([max("pm_offset").alias("max_pm_offset")])
        .collect()?
        .column("max_pm_offset")?
        .get(0)?
        .extract::<f32>()
        .unwrap_or(0.0);

    Ok(max_pm_offset * 3600.0)
}
