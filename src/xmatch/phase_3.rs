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

use crate::tools::{
    IsEmpty, PartitionedParquetSinkOptions, lf_row_count, sink_lf_to_partitioned_parquet,
    to_pl_path,
};
use crate::xmatch::models::{
    CATALOG_SCHEMA, CATALOG_TO_TABLE_SCHEMA, CatalogueModel, XMatchOptions, cast_and_reorder,
    get_column_dtype,
};
use crate::xmatch::tools::{
    add_catalogid, delete_dir_if_empty, get_hats_path_root, get_temp_dir_path,
    get_unix_timestamp_ms, get_xmatch_path_root, is_directory_empty, move_temp_files,
    propagate_coordinates,
};

pub fn run_phase_3(
    xmatch_path: PathBuf,
    cat_model: CatalogueModel,
    xmatch_options: XMatchOptions,
    move_files: bool,
) -> Result<()> {
    log::info!("Running phase 3 for catalogue {}.", cat_model.name);

    // Load the xmatch dataset.
    let catalog_dataset_path = get_xmatch_path_root(xmatch_path.clone())?.join("catalog/dataset");
    let hats_dataset_path = get_hats_path_root(cat_model.hats_path.clone())?.join("object/dataset");

    // Check that the HATS dataset exists.
    if !hats_dataset_path.exists() {
        bail!("HATS dataset not found at {}.", hats_dataset_path.display());
    }

    // Create directories if they don't exist.
    std::fs::create_dir_all(&catalog_dataset_path)?;

    let tmp_path = get_temp_dir_path(
        xmatch_path.clone(),
        &cat_model.name,
        &xmatch_options.version.clone(),
        cat_model.run_id as usize,
        true,
    )?;

    // Ensure that the max catalogid cache is update for this run_id.
    log::debug!(
        "Updating max catalogid cache for run_id {}.",
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
        log::info!("XMatch dataset is empty. Processing entire catalogue dataset.");

        let cat_lf = LazyFrame::scan_parquet(to_pl_path(hats_dataset_path)?, Default::default())?;
        _xmatch_phase3_process_lf(&cat_lf, None, &cat_model, &tmp_path, &xmatch_options)?;
    } else {
        bail!("Parallel processing of catalogue dataset is not implemented yet.");
    }

    // Log number of rows.
    let lf_catalog = LazyFrame::scan_parquet(
        to_pl_path(tmp_path.join("catalog/dataset"))?,
        Default::default(),
    )?;
    let n_rows_catalog = lf_row_count(&lf_catalog)?;

    log::info!(
        "Phase 3 for catalogue {} completed. Added {} entries to catalog.",
        cat_model.name,
        n_rows_catalog.to_formatted_string(&Locale::en)
    );

    // Copy the data from the temp path to the final xmatch path if requested.
    if move_files {
        let suffix = format!(
            "{}-{}-{}",
            cat_model.name, xmatch_options.version, cat_model.run_id
        );

        // catalog directory
        let catalog_dataset_path =
            get_xmatch_path_root(xmatch_path.clone())?.join("catalog/dataset");
        std::fs::create_dir_all(&catalog_dataset_path)?;

        let tmp_catalog_dataset_path = tmp_path.join("catalog/dataset");

        log::debug!(
            "Moving catalog dataset to {}.",
            catalog_dataset_path.display()
        );
        move_temp_files(
            tmp_catalog_dataset_path.clone(),
            catalog_dataset_path.clone(),
            Some(suffix.clone()),
        )?;

        // catalog_to_table directory
        let catalog_to_table_name = format!("catalog_to_{}", cat_model.name);

        let catalog_to_table_path = get_xmatch_path_root(xmatch_path.clone())?
            .join(format!("{}/dataset", catalog_to_table_name));
        std::fs::create_dir_all(&catalog_to_table_path)?;

        let tmp_catalog_to_table_path = tmp_path.join(format!("{}/dataset", catalog_to_table_name));

        log::debug!(
            "Moving catalog_to_table dataset to {}.",
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
            "Move files is set to false. Leaving files in temporary path {}.",
            tmp_path.display()
        );
    }

    Ok(())
}

fn _xmatch_phase3_process_lf(
    cat_lf: &LazyFrame,
    catalog_to_table_lf: Option<&LazyFrame>,
    cat_model: &CatalogueModel,
    output_path: &PathBuf,
    xmatch_options: &XMatchOptions,
) -> Result<()> {
    let cat_lf = cat_lf.clone();

    // If a catalog_to_table frame is passed, use it to discard sources that are already
    // in the xmatch dataset.
    let cat_lf = if let Some(catalog_to_table_lf) = catalog_to_table_lf {
        let primary_key = cat_model.primary_key.clone();
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
        cat_lf.clone()
    };

    // Add the catalogid column.
    let cat_lf = add_catalogid(None, &cat_lf, cat_model.run_id as usize, true)?;

    // Build the catalog lf.
    let catalog_lf = _get_catalog_lf(cat_lf, &cat_model, &xmatch_options)?;

    // Drop the pk from the catalog_lf that we'll sink but keep in the original because
    // we will need it for catalog_to_table.
    let catalog_lf_no_pk = catalog_lf.clone().drop(Selector::ByName {
        names: Arc::new(["pk".into()]),
        strict: true,
    });

    // Check that the output path is empty.
    if !is_directory_empty(&output_path) {
        bail!("Output path {} is not empty.", output_path.display());
    }

    // Sink the catalog lf to partitioned parquet.
    let catalog_path = output_path.join("catalog/dataset");

    sink_lf_to_partitioned_parquet(
        catalog_lf_no_pk,
        &catalog_path.join(format!("Norder={}", xmatch_options.pixelation_order)),
        vec!["healpix_xmatch".into()],
        PartitionedParquetSinkOptions::default(),
    )?;

    // Prepare the catalog_to_table frame.
    let primary_key = cat_model.primary_key.clone();

    let pk_dtype = get_column_dtype(&catalog_lf, "pk")?;
    let mut catalog_to_table_schema = CATALOG_TO_TABLE_SCHEMA.clone();
    catalog_to_table_schema.insert_at_index(1, primary_key.clone().into(), pk_dtype)?;

    let catalog_to_table_lf = catalog_lf.select([
        col("catalogid"),
        col("pk").alias(primary_key),
        col("healpix_29"),
        col("version"),
        lit(Null {}).cast(DataType::Float64).alias("distance"),
        lit(true).alias("best"),
        lit(3_u8).alias("added_by_phase"),
        col("added_on"),
    ]);

    let catalog_to_table_lf = cast_and_reorder(catalog_to_table_lf, &catalog_to_table_schema)?;

    // Add the healpix_xmatch column which will be used for partitioning.
    // This is the HEALPix pixel at the xmatch pixelation order.
    let catalog_to_table_lf = catalog_to_table_lf.with_column(
        (col("healpix_29")
            / lit(4_u64.pow((29 - xmatch_options.pixelation_order) as u32)).cast(DataType::UInt64))
        .alias("healpix_xmatch"),
    );

    // Sink the catalog_to_table lf to partitioned parquet.
    let catalog_to_table_name = format!("catalog_to_{}", cat_model.name);
    let catalog_to_table_path = output_path.join(format!("{}/dataset", catalog_to_table_name));

    sink_lf_to_partitioned_parquet(
        catalog_to_table_lf,
        &catalog_to_table_path.join(format!("Norder={}", xmatch_options.pixelation_order)),
        vec!["healpix_xmatch".into()],
        PartitionedParquetSinkOptions::default(),
    )?;

    Ok(())
}

fn _get_catalog_lf(
    cat_lf: LazyFrame,
    cat_model: &CatalogueModel,
    xmatch_options: &XMatchOptions,
) -> Result<LazyFrame> {
    // Initial columns for the catalog lf.
    let mut columns = vec![
        col("catalogid"),
        col(&cat_model.primary_key).alias("pk"),
        col(&cat_model.ra_column).alias("ra"),
        col(&cat_model.dec_column).alias("dec"),
        col("_healpix_29").alias("healpix_29"),
        lit(xmatch_options.version.clone()).alias("version"),
        lit(xmatch_options.epoch.clone()).alias("epoch"),
        lit(cat_model.name.clone()).alias("added_by"),
        lit(get_unix_timestamp_ms()).alias("added_on"), // Will cast to datetime.
    ];

    // pmra column
    if let Some(pmra_column) = &cat_model.pmra_column {
        columns.push(col(pmra_column).alias("pmra"))
    } else {
        columns.push(lit(Null {}).cast(DataType::Float64).alias("pmra"))
    };

    // pmdec column
    if let Some(pmdec_column) = &cat_model.pmdec_column {
        columns.push(col(pmdec_column).alias("pmdec"))
    } else {
        columns.push(lit(Null {}).cast(DataType::Float64).alias("pmdec"))
    };

    // Reference epoch column
    if let Some(ref_epoch) = &cat_model.epoch {
        columns.push(
            lit((*ref_epoch).clone())
                .cast(DataType::Float32)
                .alias("ref_epoch"),
        )
    } else {
        if let Some(ref_epoch) = &cat_model.epoch {
            columns.push(
                lit((*ref_epoch).clone())
                    .cast(DataType::Float32)
                    .alias("ref_epoch"),
            )
        } else {
            columns.push(lit(Null {}).cast(DataType::Float32).alias("ref_epoch"))
        }
    }

    // Select the columns.
    let catalog_lf = cat_lf.select(columns);

    // Propagate coordinates.
    let catalog_lf = propagate_coordinates(catalog_lf, &cat_model, xmatch_options.epoch)?;

    // Add the pk column to the catalog schema. This is not totally trivial because we
    // need to get the dtype first from the live schema.
    let mut catalog_schema_pk = CATALOG_SCHEMA.clone();
    let pk_dtype = get_column_dtype(&catalog_lf, "pk")?;
    catalog_schema_pk.insert("pk".into(), pk_dtype);

    // Cast the catalog_lf to the correct dtypes and reorder columns.
    let catalog_lf = cast_and_reorder(catalog_lf, &catalog_schema_pk)?;

    // Add the healpix_xmatch column which will be used for partitioning.
    // This is the HEALPix pixel at the xmatch pixelation order.
    let catalog_lf = catalog_lf.with_column(
        (col("healpix_29")
            / lit(4_u64.pow((29 - xmatch_options.pixelation_order) as u32)).cast(DataType::UInt64))
        .alias("healpix_xmatch"),
    );

    Ok(catalog_lf)
}
