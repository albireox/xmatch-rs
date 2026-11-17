/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-06-15
 *  @Filename: phase_3.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use std::path::PathBuf;
use std::sync::Mutex;
use std::thread::available_parallelism;

use anyhow::{Result, bail};
use log;
use num_format::{Locale, ToFormattedString};
use polars::prelude::*;

use crate::hats::dataset::{AsDataset, Dataset};
use crate::tools::{
    IsEmpty, PartitionedParquetSinkOptions, lf_row_count, sink_lf_to_partitioned_parquet,
    to_pl_path,
};
use crate::xmatch::models::{
    CATALOG_SCHEMA, CATALOG_TO_TABLE_SCHEMA, CatalogueModel, XMatchData, cast_and_reorder,
};
use crate::xmatch::tools::{
    ProgressTracker, add_catalogid, add_healpix_xmatch_column, delete_dir_if_empty,
    get_temp_dir_path, get_unix_timestamp_ms, get_xmatch_path_root, get_xmatch_pixels,
    move_temp_files, propagate_coordinates, should_skip,
};

pub(crate) fn run_phase_3(xmatch_data: XMatchData) -> Result<()> {
    let log_prefix = format!(
        "[{}] - Phase 3 -",
        xmatch_data.catalogue_name.to_uppercase()
    );

    if should_skip(&xmatch_data, 3) {
        log::info!("{} Skipping phase 3.", log_prefix);
        return Ok(());
    }

    let cat_model = xmatch_data
        .get_catalogue_model(&xmatch_data.catalogue_name)
        .unwrap();

    log::info!(
        "{} Running phase 3 for catalogue {}.",
        log_prefix,
        cat_model.name
    );

    // Get new catalogue dataset.
    let new_cat_dataset = Dataset::new(cat_model.hats_path.clone(), true);

    // Check that the HATS dataset exists.
    if !new_cat_dataset.path().exists() {
        bail!(
            "{} HATS dataset not found at {}.",
            log_prefix,
            new_cat_dataset.path().display()
        );
    }

    // Create directories if they don't exist.
    let catalog_dataset_path =
        get_xmatch_path_root(xmatch_data.path.clone())?.join("catalog/dataset");
    std::fs::create_dir_all(&catalog_dataset_path)?;

    let tmp_path = get_temp_dir_path(
        xmatch_data.path.clone(),
        &cat_model.name,
        &xmatch_data.model.version.clone(),
        cat_model.run_id as usize,
        true,
    )?;

    // Ensure that the max catalogid cache is updated for this run_id.
    log::debug!(
        "{} Updating max. catalogid cache for run_id {}.",
        log_prefix,
        cat_model.run_id
    );

    crate::xmatch::tools::get_max_catalogid(
        Some(catalog_dataset_path.clone()),
        Some(cat_model.run_id as usize),
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
            log_prefix
        );

        let new_cat_lf = new_cat_dataset.get_frame()?;

        xmatch_phase3_process_lf(&new_cat_lf, None, &xmatch_data, &cat_model, &tmp_path)?;
    } else {
        // Use a quarter of the CPUs if n_workers is not specified.
        let phase = cat_model.get_phase(3).unwrap();
        let n_workers = phase
            .n_workers
            .unwrap_or_else(|| available_parallelism().unwrap().get() / 4);

        log::info!(
            "{} Processing catalogue dataset in parallel with {} workers.",
            log_prefix,
            n_workers
        );

        // Parallelise the processing of each pixel in the new catalogue.
        let new_cat_partition_info = new_cat_dataset.partition();

        // Get a progress tracker.
        let progress_tracker = if let Some(progress_file) = &xmatch_data.options.progress_file {
            Some(ProgressTracker::new(
                progress_file.clone(),
                new_cat_partition_info.len() as i32,
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

            new_cat_partition_info
                .into_iter()
                .par_bridge()
                .try_for_each(|(order, pixel)| {
                    // Get the catalogue pixel from the HATS dataset.
                    let cat_pixel_lf = new_cat_dataset.scan_pixel(order, pixel, false)?;

                    // If the pixel is empty (can this happen?) then just return.
                    if cat_pixel_lf.is_empty() {
                        return Ok(());
                    }

                    // Find the associated pixels in the xmatch dataset. We include neighbours because
                    // the match may be outside the nominal pixel due to proper motion propagation.
                    // We create a vector of tuples with the pixelation order as first element.
                    // TODO: can this be replaced with the Dataset.get_neighbours() method?
                    let xmatch_pixels =
                        get_xmatch_pixels(xmatch_data.model.pixelation_order, order, pixel, true)?
                            .iter()
                            .map(|pp| (xmatch_data.model.pixelation_order, *pp))
                            .collect::<Vec<_>>();

                    // Scan the relevant pixels.
                    let xmatch_dataset_path = get_xmatch_path_root(xmatch_data.path.clone())?;

                    let catalog_to_table_path =
                        xmatch_dataset_path.join(format!("catalog_to_{}/dataset", cat_model.name));

                    let catalog_to_table_lf = if catalog_to_table_path.exists() {
                        let catalog_to_table_dataset =
                            Dataset::new(catalog_to_table_path.clone(), false);
                        let catalog_to_table_partition = catalog_to_table_dataset.partition();
                        let cat_to_pixels = xmatch_pixels
                            .iter()
                            .filter(|(oo, pp)| {
                                catalog_to_table_partition.contains(*oo, *pp).unwrap()
                            })
                            .copied()
                            .collect::<Vec<_>>();

                        if cat_to_pixels.is_empty() {
                            return Ok(());
                        }

                        catalog_to_table_dataset.scan_pixels(cat_to_pixels)?
                    } else {
                        let pk_dtype = cat_pixel_lf
                            .clone()
                            .collect_schema()
                            .unwrap()
                            .get(&cat_model.primary_key.clone())
                            .unwrap()
                            .clone();

                        let catalog_to_table_df =
                            DataFrame::empty_with_arc_schema(Arc::new(Schema::from_iter(vec![
                                Field::new("catalogid".into(), DataType::UInt64),
                                Field::new(cat_model.primary_key.clone().into(), pk_dtype),
                            ])));

                        catalog_to_table_df.lazy()
                    };

                    xmatch_phase3_process_lf(
                        &cat_pixel_lf,
                        Some(&catalog_to_table_lf),
                        &xmatch_data,
                        &cat_model,
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
        log_prefix,
        cat_model.name,
        n_rows_catalog.to_formatted_string(&Locale::en)
    );

    // Copy the data from the temp path to the final xmatch path if requested.
    if xmatch_data.options.move_files {
        let suffix = format!(
            "{}-{}-{}",
            cat_model.name, xmatch_data.model.version, cat_model.run_id
        );

        // catalog directory
        let catalog_dataset_path =
            get_xmatch_path_root(xmatch_data.path.clone())?.join("catalog/dataset");
        std::fs::create_dir_all(&catalog_dataset_path)?;

        let tmp_catalog_dataset_path = tmp_path.join("catalog/dataset");

        log::debug!(
            "{} Moving catalog dataset to {}.",
            log_prefix,
            catalog_dataset_path.display()
        );
        move_temp_files(
            tmp_catalog_dataset_path.clone(),
            catalog_dataset_path.clone(),
            Some(suffix.clone()),
        )?;

        // catalog_to_table directory
        let catalog_to_table_name = format!("catalog_to_{}", cat_model.name);

        let catalog_to_table_path = get_xmatch_path_root(xmatch_data.path.clone())?
            .join(format!("{}/dataset", catalog_to_table_name));
        std::fs::create_dir_all(&catalog_to_table_path)?;

        let tmp_catalog_to_table_path = tmp_path.join(format!("{}/dataset", catalog_to_table_name));

        log::debug!(
            "{} Moving catalog_to_table dataset to {}.",
            log_prefix,
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
            log_prefix,
            tmp_path.display()
        );
    }

    Ok(())
}

// Lock to prevent multiple threads from reading/updating the catalogid cache at the same time.
static CATID_LOCK: Mutex<()> = Mutex::new(());

fn xmatch_phase3_process_lf(
    cat_lf: &LazyFrame,
    catalog_to_table_lf: Option<&LazyFrame>,
    xmatch_data: &XMatchData,
    cat_model: &CatalogueModel,
    output_path: &PathBuf,
) -> Result<()> {
    let primary_key = cat_model.primary_key.clone();

    let cat_lf = cat_lf.clone();

    // If a catalog_to_table frame is passed, use it to discard sources that are already
    // in the xmatch dataset.
    let mut cat_lf = cat_lf.clone();

    if let Some(catalog_to_table_lf) = catalog_to_table_lf {
        if !catalog_to_table_lf.is_empty() {
            let primary_key = cat_model.primary_key.clone();
            let catalog_to_table_lf = catalog_to_table_lf.clone().select([col(&primary_key)]);

            cat_lf = cat_lf.join(
                catalog_to_table_lf,
                [col(&primary_key)],
                [col(&primary_key)],
                JoinArgs {
                    how: JoinType::Anti,
                    ..Default::default()
                },
            )
        }
    }

    if cat_lf.is_empty() {
        return Ok(());
    }

    // Add the catalogid column. Here we rely on the catalogid cache. To prevent multiple threads
    // reading/updating the cache at the same time, we use a mutex lock.
    let _guard = CATID_LOCK.lock().unwrap();
    let cat_lf = add_catalogid(None, &cat_lf, cat_model.run_id as usize, true)?;
    drop(_guard);

    // Build the catalog lf.
    let catalog_lf = get_catalog_lf(cat_lf, &xmatch_data, &cat_model)?;

    // Drop the pk from the catalog_lf that we'll sink but keep in the original because
    // we will need it for catalog_to_table.
    let catalog_lf_no_pk = catalog_lf.clone().drop(Selector::ByName {
        names: Arc::new(["pk".into()]),
        strict: true,
    });

    // Cast the catalog_lf to the correct dtypes and reorder columns.
    let catalog_lf_no_pk = cast_and_reorder(catalog_lf_no_pk, &CATALOG_SCHEMA, None)?;
    let catalog_lf_no_pk =
        add_healpix_xmatch_column(catalog_lf_no_pk, xmatch_data.model.pixelation_order)?;

    // Sink the catalog lf to partitioned parquet.
    let catalog_path = output_path.join("catalog/dataset");

    // This sink partitions on healpix_xmatch but the file path provider in
    // PartitionedParquetSinkOptions actually creates the Dir=/Npix= directories.
    sink_lf_to_partitioned_parquet(
        catalog_lf_no_pk,
        &catalog_path.join(format!("Norder={}", xmatch_data.model.pixelation_order)),
        vec!["healpix_xmatch".into()],
        PartitionedParquetSinkOptions::default(),
    )?;

    // Prepare the catalog_to_table frame.
    let catalog_to_table_lf = catalog_lf.select([
        col("catalogid"),
        col("pk"),
        col("_healpix_29"),
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
        add_healpix_xmatch_column(catalog_to_table_lf, xmatch_data.model.pixelation_order)?;

    // Sink the catalog_to_table lf to partitioned parquet.
    let catalog_to_table_name = format!("catalog_to_{}", cat_model.name);
    let catalog_to_table_path = output_path.join(format!("{}/dataset", catalog_to_table_name));

    sink_lf_to_partitioned_parquet(
        catalog_to_table_lf,
        &catalog_to_table_path.join(format!("Norder={}", xmatch_data.model.pixelation_order)),
        vec!["healpix_xmatch".into()],
        PartitionedParquetSinkOptions::default(),
    )?;

    Ok(())
}

fn get_catalog_lf(
    cat_lf: LazyFrame,
    xmatch_data: &XMatchData,
    cat_model: &CatalogueModel,
) -> Result<LazyFrame> {
    // Add empty pmra column if the catalogue does not have one.
    let cat_lf = if let None = &cat_model.pmra_column {
        cat_lf.with_column(lit(Null {}).cast(DataType::Float64).alias("pmra"))
    } else {
        cat_lf
    };

    // pmdec column
    let cat_lf = if let None = &cat_model.pmdec_column {
        cat_lf.with_column(lit(Null {}).cast(DataType::Float64).alias("pmdec"))
    } else {
        cat_lf
    };

    // Propagate coordinates.
    let catalog_lf = propagate_coordinates(cat_lf, &cat_model, xmatch_data.model.epoch)?;

    // If we have propagated the coordinats to the xmatch epoch, then the _healpix_29 column is
    // no longer valid and we need to recalculate it.
    let catalog_lf = if cat_model.pmra_column.is_none()
        || cat_model.pmdec_column.is_none()
        || (cat_model.epoch.is_none() && cat_model.epoch_column.is_none())
    {
        catalog_lf
    } else {
        catalog_lf
            .drop(by_name(["_healpix_29"], true, false))
            .with_column(
                map_multiple(
                    |columns| {
                        // Create a layer for the order 29 healpix pixelation.
                        let layer = cdshealpix::nested::get(29);

                        // Extract the two Series/Columns for ra and dec.
                        let ra = columns[0].f64()?;
                        let dec = columns[1].f64()?;

                        // Perform custom elementwise logic using ChunkedArray iterators
                        let result: UInt64Chunked = ra
                            .iter()
                            .zip(dec.iter())
                            .map(|(opt_ra, opt_dec)| match (opt_ra, opt_dec) {
                                (Some(val_ra), Some(val_dec)) => {
                                    Some(layer.hash(val_ra.to_radians(), val_dec.to_radians())
                                        as u64)
                                }
                                _ => None,
                            })
                            .collect();

                        Ok(result.into_column())
                    },
                    // Input expressions
                    &[col("ra_epoch"), col("dec_epoch")],
                    // Output type specification for the query planner
                    |_schema, _fields| Ok(Field::new("_healpix_29".into(), DataType::UInt64)),
                )
                .alias("custom_result"),
            )
    };

    // Select columns from the catalog lf.
    let mut columns = vec![
        col("catalogid"),
        col(&cat_model.primary_key).alias("pk"),
        col("_healpix_29"),
        col("ra_epoch").alias("ra"),
        col("dec_epoch").alias("dec"),
        lit(xmatch_data.model.version.clone()).alias("version"),
        lit(xmatch_data.model.epoch.clone()).alias("epoch"),
        lit(cat_model.name.clone()).alias("added_by"),
        lit(get_unix_timestamp_ms()).alias("added_on"), // Will cast to datetime.
    ];

    if let Some(pmra_column) = &cat_model.pmra_column {
        columns.push(col(pmra_column).alias("pmra"));
    } else {
        columns.push(col("pmra"));
    }

    if let Some(pmdec_column) = &cat_model.pmdec_column {
        columns.push(col(pmdec_column).alias("pmdec"));
    } else {
        columns.push(col("pmdec"));
    }

    let catalog_lf = catalog_lf.select(columns);

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
