/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-05-18
 *  @Filename: hats.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use std::path::PathBuf;

use anyhow::{Result, bail};
use pyo3::prelude::*;

use crate::hats::{
    self,
    margin::{CellSide, PointsStrategy, path_along_edge},
};

#[pyfunction(name = "repartition")]
pub(crate) fn hats_repartition(
    path: PathBuf,
    order_0: u8,
    order_1: u8,
    pixels: Option<Vec<i64>>,
    max_rows_per_pixel: Option<usize>,
    max_kilobytes_per_pixel: Option<u64>,
    n_workers: Option<usize>,
    progress_file: Option<PathBuf>,
    keep_initial_order_directories: bool,
) -> Result<Vec<i64>> {
    let options = hats::partition::HATSRepartitionOptions {
        max_rows_per_pixel,
        max_kilobytes_per_pixel,
        n_workers,
        progress_file,
        keep_initial_order_directories,
        ..hats::partition::HATSRepartitionOptions::default()
    };

    hats::partition::hats_repartition(path, order_0, order_1, pixels, options)
}

#[pyfunction(name = "rechunk")]
pub(crate) fn hats_rechunk(
    path: PathBuf,
    is_dataset: bool,
    max_rows_per_file: Option<u32>,
    approximate_bytes_per_file: Option<u64>,
) -> Result<()> {
    let options = hats::rechunk::HATSRechunkOptions {
        is_dataset,
        max_rows_per_file,
        approximate_bytes_per_file,
        sort_by: None,
    };

    hats::rechunk::rechunk_hats_dataset(path, options)
}

#[pyfunction(name = "create_index")]
pub(crate) fn hats_create_index(
    collection_path: PathBuf,
    index_name: &str,
    index_column: &str,
    extra_columns: Option<Vec<String>>,
    mode: Option<String>,
    check_duplicates: Option<bool>,
    max_rows_per_file: Option<usize>,
    max_kilobytes_per_file: Option<u64>,
) -> Result<()> {
    let def_opts = hats::index::HATSCreateIndexOptions::default();

    let options = hats::index::HATSCreateIndexOptions {
        mode: mode.unwrap_or_else(|| def_opts.mode.clone()),
        check_duplicates: check_duplicates.unwrap_or(def_opts.check_duplicates),
        max_rows_per_file: max_rows_per_file.unwrap_or(def_opts.max_rows_per_file),
        max_kilobytes_per_file: max_kilobytes_per_file.unwrap_or(def_opts.max_kilobytes_per_file),
    };

    hats::index::create_hats_index(
        collection_path,
        index_name,
        index_column,
        extra_columns,
        options,
    )?;

    Ok(())
}

#[pyfunction(name = "get_neighbours")]
pub(crate) fn hats_get_neighbours(
    dataset_path: PathBuf,
    norder: u8,
    npix: u64,
) -> Result<Vec<(u8, u64)>> {
    hats::tools::get_hats_neighbours(dataset_path, norder, npix, None)
}

#[pyfunction(name = "create_margin_cache")]
pub(crate) fn hats_create_margin_cache(
    collection_path: PathBuf,
    margin_name: &str,
    buffer: f32,
    overwrite: bool,
    n_workers: Option<usize>,
    progress_file: Option<PathBuf>,
) -> Result<()> {
    let options = hats::margin::MarginCacheOptions {
        n_workers,
        progress_file,
        overwrite,
    };

    hats::margin::create_margin_cache(&collection_path, margin_name, buffer, options)
}

#[pyfunction(name = "path_along_edge")]
pub(crate) fn py_path_along_edge(
    norder: u8,
    npix: u64,
    grow: Option<f32>,
    side: &str,
) -> Result<Vec<(f64, f64)>> {
    let cell_side = match side {
        "NW" => CellSide::NW,
        "SW" => CellSide::SW,
        "NE" => CellSide::NE,
        "SE" => CellSide::SE,
        _ => bail!(format!(
            "Invalid side '{}'. Valid options are 'NW', 'NE', 'SW', 'SE'.",
            side
        )),
    };

    Ok(path_along_edge(
        norder,
        npix,
        cell_side,
        grow,
        true,
        PointsStrategy::Dynamic,
        true,
    ))
}
