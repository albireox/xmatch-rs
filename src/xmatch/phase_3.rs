/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-06-15
 *  @Filename: phase_3.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use anyhow::{Result, bail};
use log;
use num_format::{Locale, ToFormattedString};
use polars::prelude::*;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::thread::available_parallelism;

use crate::hats::tools::PartitionInfo;
use crate::tools::{
    IsEmpty, PartitionedParquetSinkOptions, lf_row_count, sink_lf_to_partitioned_parquet,
    to_pl_path,
};
use crate::xmatch::models::{
    CATALOG_SCHEMA, CATALOG_TO_TABLE_SCHEMA, CatalogueModel, XMatchModel, XMatchOptions,
    cast_and_reorder,
};
use crate::xmatch::tools::{
    ProgressTracker, add_catalogid, add_healpix_xmatch_column, delete_dir_if_empty,
    get_hats_path_root, get_temp_dir_path, get_unix_timestamp_ms, get_xmatch_path_root,
    get_xmatch_pixels, move_temp_files, propagate_coordinates, scan_pixel_from_hats_dataset,
    scan_pixels_from_xmatch_dataset,
};

static LOG_PREFIX: OnceLock<String> = OnceLock::new();

pub fn run_phase_3(
    xmatch_path: PathBuf,
    xmatch_data: XMatchModel,
    cat_data: CatalogueModel,
    xmatch_options: XMatchOptions,
) -> Result<()> {
    LOG_PREFIX.get_or_init(|| format!("[{}] - Phase 3 -", cat_data.name.to_uppercase()));

    log::info!(
        "{} Running phase 3 for catalogue {}.",
        LOG_PREFIX.get().unwrap(),
        cat_data.name
    );

    // Get dataset paths.
    let catalog_dataset_path = get_xmatch_path_root(xmatch_path.clone())?.join("catalog/dataset");
    let hats_dataset_path = get_hats_path_root(cat_data.hats_path.clone())?.join("object/dataset");

    // Check that the HATS dataset exists.
    if !hats_dataset_path.exists() {
        bail!(
            "{} HATS dataset not found at {}.",
            LOG_PREFIX.get().unwrap(),
            hats_dataset_path.display()
        );
    }

    // Create directories if they don't exist.
    std::fs::create_dir_all(&catalog_dataset_path)?;

    let tmp_path = get_temp_dir_path(
        xmatch_path.clone(),
        &cat_data.name,
        &xmatch_data.version.clone(),
        cat_data.run_id as usize,
        true,
    )?;

    // Ensure that the max catalogid cache is update for this run_id.
    log::debug!(
        "{} Updating max catalogid cache for run_id {}.",
        LOG_PREFIX.get().unwrap(),
        cat_data.run_id
    );

    crate::xmatch::tools::get_max_catalogid(
        Some(catalog_dataset_path.clone()),
        Some(cat_data.run_id as usize),
        true,
    )?;

    let xmatch_lf = LazyFrame::scan_parquet(
        to_pl_path(catalog_dataset_path.clone())?,
        ScanArgsParquet {
            schema: Some(Arc::new(CATALOG_SCHEMA.clone())),
            ..Default::default()
        },
    )?;

    // If the dataset is empty, this means this is the first time we run phase 3 (first catalogue).
    // In this case it's more efficient to not parallelise and just let Polars do its thing.
    if xmatch_lf.is_empty() {
        log::info!(
            "{} XMatch dataset is empty. Processing entire catalogue dataset. No progress bar will be shown.",
            LOG_PREFIX.get().unwrap()
        );

        let cat_lf = LazyFrame::scan_parquet(to_pl_path(hats_dataset_path)?, Default::default())?;
        _xmatch_phase3_process_lf(&cat_lf, None, &xmatch_data, &cat_data, &tmp_path)?;
    } else {
        // Use a quarter of the CPUs if n_workers is not specified.
        let phase = cat_data.get_phase(3).unwrap();
        let n_workers = phase
            .n_workers
            .unwrap_or_else(|| available_parallelism().unwrap().get() / 4);

        log::info!(
            "{} Processing catalogue dataset in parallel with {} workers.",
            LOG_PREFIX.get().unwrap(),
            n_workers
        );

        // Parallelise the processing of each pixel. For each pixel in the catalogue dataset, we
        // find the associated pixels in the catalog_to_table dataset and reject entries that are
        // already there (added by phases 1 or 2).
        let partition_info = PartitionInfo::from_dataset(&hats_dataset_path, true)?;

        // Get a progress tracker.
        let progress_tracker = if let Some(progress_file) = &xmatch_options.progress_file {
            Some(ProgressTracker::new(
                progress_file.clone(),
                partition_info.len() as i32,
                true,
            ))
        } else {
            None
        };

        let progress_tracker = Mutex::new(progress_tracker);

        let pool = crate::tools::create_pool(Some(n_workers))?;
        pool.install(|| {
            use rayon::iter::ParallelIterator;
            use rayon::prelude::*;

            partition_info
                .into_iter()
                .par_bridge()
                .try_for_each(|(order, pixel)| {
                    // Get the catalogue pixel from the HATS dataset.
                    let cat_pixel_lf = scan_pixel_from_hats_dataset(
                        hats_dataset_path.clone(),
                        order,
                        pixel,
                        true,
                    )?;

                    let cat_pixel = if let Some(cat_pixel_lf) = cat_pixel_lf {
                        cat_pixel_lf
                    } else {
                        return Ok(());
                    };

                    // Find the associated pixels in the xmatch dataset. We include neighbours because
                    // the match may be outside the nominal pixel due to proper motion propagation.
                    let xmatch_pixels =
                        get_xmatch_pixels(xmatch_data.pixelation_order, order, pixel, true)?;

                    // Scan the relevant pixels.
                    let xmatch_dataset_path = get_xmatch_path_root(xmatch_path.clone())?;

                    let catalog_to_table_path =
                        xmatch_dataset_path.join(format!("catalog_to_{}/dataset", cat_data.name));

                    let catalog_to_table_pixel = if catalog_to_table_path.exists() {
                        scan_pixels_from_xmatch_dataset(
                            xmatch_dataset_path.clone(),
                            Some(cat_data.name.clone()),
                            xmatch_data.pixelation_order,
                            &xmatch_pixels,
                        )?
                    } else {
                        let pk_dtype = cat_pixel
                            .clone()
                            .collect_schema()
                            .unwrap()
                            .get(&cat_data.primary_key.clone())
                            .unwrap()
                            .clone();

                        let catalog_to_table_df =
                            DataFrame::empty_with_arc_schema(Arc::new(Schema::from_iter(vec![
                                Field::new("catalogid".into(), DataType::UInt64),
                                Field::new(cat_data.primary_key.clone().into(), pk_dtype),
                            ])));

                        catalog_to_table_df.lazy()
                    };

                    _xmatch_phase3_process_lf(
                        &cat_pixel,
                        Some(&catalog_to_table_pixel),
                        &xmatch_data,
                        &cat_data,
                        &tmp_path,
                    )?;

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
    }

    // Log number of rows.
    let lf_catalog = LazyFrame::scan_parquet(
        to_pl_path(tmp_path.join("catalog/dataset"))?,
        Default::default(),
    )?;
    let n_rows_catalog = lf_row_count(&lf_catalog)?;

    log::info!(
        "{} Phase 3 for catalogue {} completed. Added {} entries to catalog.",
        LOG_PREFIX.get().unwrap(),
        cat_data.name,
        n_rows_catalog.to_formatted_string(&Locale::en)
    );

    // Copy the data from the temp path to the final xmatch path if requested.
    if xmatch_options.move_files {
        let suffix = format!(
            "{}-{}-{}",
            cat_data.name, xmatch_data.version, cat_data.run_id
        );

        // catalog directory
        let catalog_dataset_path =
            get_xmatch_path_root(xmatch_path.clone())?.join("catalog/dataset");
        std::fs::create_dir_all(&catalog_dataset_path)?;

        let tmp_catalog_dataset_path = tmp_path.join("catalog/dataset");

        log::debug!(
            "{} Moving catalog dataset to {}.",
            LOG_PREFIX.get().unwrap(),
            catalog_dataset_path.display()
        );
        move_temp_files(
            tmp_catalog_dataset_path.clone(),
            catalog_dataset_path.clone(),
            Some(suffix.clone()),
        )?;

        // catalog_to_table directory
        let catalog_to_table_name = format!("catalog_to_{}", cat_data.name);

        let catalog_to_table_path = get_xmatch_path_root(xmatch_path.clone())?
            .join(format!("{}/dataset", catalog_to_table_name));
        std::fs::create_dir_all(&catalog_to_table_path)?;

        let tmp_catalog_to_table_path = tmp_path.join(format!("{}/dataset", catalog_to_table_name));

        log::debug!(
            "{} Moving catalog_to_table dataset to {}.",
            LOG_PREFIX.get().unwrap(),
            catalog_to_table_path.display()
        );
        move_temp_files(
            tmp_catalog_to_table_path.clone(),
            catalog_to_table_path.clone(),
            Some(suffix.clone()),
        )?;

        delete_dir_if_empty(tmp_path.parent().unwrap())?;
    } else {
        log::info!(
            "{} Move files is set to false. Leaving files in temporary path {}.",
            LOG_PREFIX.get().unwrap(),
            tmp_path.display()
        );
    }

    Ok(())
}

fn _xmatch_phase3_process_lf(
    cat_lf: &LazyFrame,
    catalog_to_table_lf: Option<&LazyFrame>,
    xmatch_data: &XMatchModel,
    cat_data: &CatalogueModel,
    output_path: &PathBuf,
) -> Result<()> {
    let primary_key = cat_data.primary_key.clone();

    let cat_lf = cat_lf.clone();

    // If a catalog_to_table frame is passed, use it to discard sources that are already
    // in the xmatch dataset.
    let cat_lf = if let Some(catalog_to_table_lf) = catalog_to_table_lf {
        let primary_key = cat_data.primary_key.clone();
        let catalog_to_table_lf = catalog_to_table_lf.clone().select([col(&primary_key)]);

        cat_lf.join(
            catalog_to_table_lf,
            [col(&primary_key)],
            [col(&primary_key)],
            JoinArgs {
                how: JoinType::Anti,
                ..Default::default()
            },
        )
    } else {
        cat_lf
    };

    // Add the catalogid column.
    let cat_lf = add_catalogid(None, &cat_lf, cat_data.run_id as usize, true)?;

    // Build the catalog lf.
    let catalog_lf = _get_catalog_lf(cat_lf, &xmatch_data, &cat_data)?;

    // Drop the pk from the catalog_lf that we'll sink but keep in the original because
    // we will need it for catalog_to_table.
    let catalog_lf_no_pk = catalog_lf.clone().drop(Selector::ByName {
        names: Arc::new(["pk".into()]),
        strict: true,
    });

    // Cast the catalog_lf to the correct dtypes and reorder columns.
    let catalog_lf_no_pk = cast_and_reorder(catalog_lf_no_pk, &CATALOG_SCHEMA, None)?;
    let catalog_lf_no_pk =
        add_healpix_xmatch_column(catalog_lf_no_pk, xmatch_data.pixelation_order)?;

    // Sink the catalog lf to partitioned parquet.
    let catalog_path = output_path.join("catalog/dataset");

    sink_lf_to_partitioned_parquet(
        catalog_lf_no_pk,
        &catalog_path.join(format!("Norder={}", xmatch_data.pixelation_order)),
        vec!["healpix_xmatch".into()],
        PartitionedParquetSinkOptions::default(),
    )?;

    // Prepare the catalog_to_table frame.
    let catalog_to_table_lf = catalog_lf.select([
        col("catalogid"),
        col("pk"),
        col("healpix_29"),
        col("version"),
        lit(Null {}).cast(DataType::Float64).alias("distance"),
        lit(true).alias("best"),
        lit(3_u8).alias("added_by_phase"),
        col("added_on"),
    ]);

    let catalog_to_table_lf = cast_and_reorder(
        catalog_to_table_lf,
        &CATALOG_TO_TABLE_SCHEMA,
        Some(primary_key.clone()),
    )?;

    // Add the healpix_xmatch column which will be used for partitioning.
    let catalog_to_table_lf =
        add_healpix_xmatch_column(catalog_to_table_lf, xmatch_data.pixelation_order)?;

    // Sink the catalog_to_table lf to partitioned parquet.
    let catalog_to_table_name = format!("catalog_to_{}", cat_data.name);
    let catalog_to_table_path = output_path.join(format!("{}/dataset", catalog_to_table_name));

    sink_lf_to_partitioned_parquet(
        catalog_to_table_lf,
        &catalog_to_table_path.join(format!("Norder={}", xmatch_data.pixelation_order)),
        vec!["healpix_xmatch".into()],
        PartitionedParquetSinkOptions::default(),
    )?;

    Ok(())
}

fn _get_catalog_lf(
    cat_lf: LazyFrame,
    xmatch_data: &XMatchModel,
    cat_data: &CatalogueModel,
) -> Result<LazyFrame> {
    // Initial columns for the catalog lf.
    let mut columns = vec![
        col("catalogid"),
        col(&cat_data.primary_key).alias("pk"),
        col(&cat_data.ra_column).alias("ra"),
        col(&cat_data.dec_column).alias("dec"),
        col("_healpix_29").alias("healpix_29"),
        lit(xmatch_data.version.clone()).alias("version"),
        lit(xmatch_data.epoch.clone()).alias("epoch"),
        lit(cat_data.name.clone()).alias("added_by"),
        lit(get_unix_timestamp_ms()).alias("added_on"), // Will cast to datetime.
    ];

    // pmra column
    if let Some(pmra_column) = &cat_data.pmra_column {
        columns.push(col(pmra_column).alias("pmra"))
    } else {
        columns.push(lit(Null {}).cast(DataType::Float64).alias("pmra"))
    };

    // pmdec column
    if let Some(pmdec_column) = &cat_data.pmdec_column {
        columns.push(col(pmdec_column).alias("pmdec"))
    } else {
        columns.push(lit(Null {}).cast(DataType::Float64).alias("pmdec"))
    };

    // Reference epoch column
    if let Some(ref_epoch) = &cat_data.epoch {
        columns.push(
            lit((*ref_epoch).clone())
                .cast(DataType::Float32)
                .alias("ref_epoch"),
        )
    } else {
        columns.push(lit(Null {}).cast(DataType::Float32).alias("ref_epoch"))
    }

    // Select the columns.
    let catalog_lf = cat_lf.select(columns);

    // Propagate coordinates.
    let catalog_lf = propagate_coordinates(catalog_lf, &cat_data, xmatch_data.epoch)?;

    // Just in case, remove rows without valid RA/Dec.
    let null_selector = Some(Selector::ByName {
        names: Arc::new(["catalogid".into(), "ra".into(), "dec".into()]),
        strict: true,
    });

    let catalog_lf = catalog_lf
        .drop_nans(null_selector.clone())
        .drop_nulls(null_selector.clone());

    Ok(catalog_lf)
}
