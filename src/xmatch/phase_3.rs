/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-06-15
 *  @Filename: phase_3.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use anyhow::{Result, bail};
use log;
use polars::prelude::*;
use std::path::PathBuf;

use crate::tools::{
    IsEmpty, PartitionedParquetSinkOptions, sink_lf_to_partitioned_parquet, to_pl_path,
};
use crate::xmatch::models::{CATALOG_SCHEMA, CatalogueModel, XMatchOptions, cast_and_reorder};
use crate::xmatch::tools::{
    add_catalogid, get_hats_path_root, get_unix_timestamp_ms, get_xmatch_path_root,
    is_directory_empty,
};

pub fn run_phase_3(
    xmatch_path: PathBuf,
    cat_model: CatalogueModel,
    output_path: PathBuf,
    xmatch_options: XMatchOptions,
) -> Result<()> {
    // Load the xmatch dataset.
    let catalog_dataset_path = get_xmatch_path_root(xmatch_path.clone())?.join("catalog/dataset");
    let hats_dataset_path = get_hats_path_root(cat_model.hats_path.clone())?.join("object/dataset");

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
        to_pl_path(catalog_dataset_path)?,
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
        _xmatch_phase3_process_lf(cat_lf, None, cat_model, output_path, xmatch_options)?;
    }

    Ok(())
}

fn _xmatch_phase3_process_lf(
    cat_lf: LazyFrame,
    xmatch_lf: Option<LazyFrame>,
    cat_model: CatalogueModel,
    output_path: PathBuf,
    xmatch_options: XMatchOptions,
) -> Result<()> {
    if let Some(_) = xmatch_lf {
        bail!("Phase 3 for non-initial catalogues is not yet implemented.");
    };

    // Add the catalogid column.
    let cat_lf = add_catalogid(None, cat_lf, cat_model.run_id as usize)?;

    // Build the catalog lf.
    let catalog_lf = _get_catalog_lf(cat_lf, cat_model, xmatch_options)?;

    // Check that the output path is empty.
    if !is_directory_empty(&output_path) {
        bail!("Output path {} is not empty.", output_path.display());
    }

    let catalog_path = output_path.join("catalog/dataset");
    log::info!("Writing catalog output to {}.", catalog_path.display());

    sink_lf_to_partitioned_parquet(
        catalog_lf,
        &catalog_path,
        vec!["healpix_xmatch".into()],
        PartitionedParquetSinkOptions::default(),
    )?;

    Ok(())
}

fn _get_catalog_lf(
    cat_lf: LazyFrame,
    cat_model: CatalogueModel,
    xmatch_options: XMatchOptions,
) -> Result<LazyFrame> {
    // Initial columns for the catalog lf.
    let mut columns = vec![
        col("catalogid"),
        col(cat_model.ra_column).alias("ra"),
        col(cat_model.dec_column).alias("dec"),
        col("_healpix_29").alias("healpix_29"),
        lit(xmatch_options.version).alias("version"),
        lit(xmatch_options.epoch).alias("epoch"),
        lit(cat_model.name).alias("added_by"),
        lit(get_unix_timestamp_ms()).alias("added_on"), // Will cast to datetime.
    ];

    // pmra column
    if let Some(pmra_column) = cat_model.pmra_column {
        columns.push(col(pmra_column).alias("pmra"))
    } else {
        columns.push(lit(Null {}).cast(DataType::Float64).alias("pmra"))
    };

    // pmdec column
    if let Some(pmdec_column) = cat_model.pmdec_column {
        columns.push(col(pmdec_column).alias("pmdec"))
    } else {
        columns.push(lit(Null {}).cast(DataType::Float64).alias("pmdec"))
    };

    // Select the columns.
    let catalog_lf = cat_lf.select(columns);

    // Cast the catalog_lf to the correct dtypes and reorder columns.
    let catalog_lf = cast_and_reorder(catalog_lf, &CATALOG_SCHEMA)?;

    // Add the healpix_xmatch column which will be used for partitioning.
    // This is the HEALPix pixel at the xmatch pixelation order.
    let catalog_lf = catalog_lf.with_column(
        (col("healpix_29")
            / lit(4_u64.pow((29 - xmatch_options.pixelation_order) as u32)).cast(DataType::UInt64))
        .alias("healpix_xmatch"),
    );

    Ok(catalog_lf)
}
