/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-05-16
 *  @Filename: python.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use std::path::PathBuf;

use anyhow::Result;
use pyo3::{prelude::*, py_run};

use crate::{hats, healpix};

/// Converts RA and Dec to HEALPix pixel.
#[pyfunction]
fn ra_dec_to_healpix(ra: f64, dec: f64, order: u8) -> PyResult<u64> {
    Ok(healpix::ra_dec_to_healpix(ra, dec, order))
}

/// Returns the area of a HEALPix pixel at the given order in square degrees or steradians.
#[pyfunction]
fn healpix_pixel_area(order: u8, degrees: bool) -> PyResult<f64> {
    Ok(healpix::healpix_pixel_area(order, degrees))
}

#[pyfunction]
fn hats_repartition(
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
    let options = hats::HATSRepartitionOptions {
        max_rows_per_pixel,
        max_kilobytes_per_pixel,
        n_workers,
        progress_file,
        keep_initial_order_directories,
        ..hats::HATSRepartitionOptions::default()
    };

    hats::hats_repartition(path, order_0, order_1, pixels, options)
}

// Define the Python module
#[pymodule(name = "_xmatch_rs_lib")]
fn xmatch_rs_pymodule(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // xmatch_rs.healpix module
    let xmatch_rs_healpix = PyModule::new(_py, "healpix")?;
    xmatch_rs_healpix.add_function(wrap_pyfunction!(ra_dec_to_healpix, &xmatch_rs_healpix)?)?;
    xmatch_rs_healpix.add_function(wrap_pyfunction!(healpix_pixel_area, &xmatch_rs_healpix)?)?;
    py_run!(
        _py,
        xmatch_rs_healpix,
        "import sys; sys.modules['xmatch_rs._xmatch_rs_lib.healpix'] = xmatch_rs_healpix"
    );
    m.add_submodule(&xmatch_rs_healpix)?;

    // xmatch_rs.hats module
    let xmatch_rs_hats = PyModule::new(_py, "hats")?;
    xmatch_rs_hats.add_function(wrap_pyfunction!(hats_repartition, &xmatch_rs_hats)?)?;
    py_run!(
        _py,
        xmatch_rs_hats,
        "import sys; sys.modules['xmatch_rs._xmatch_rs_lib.hats'] = xmatch_rs_hats"
    );
    m.add_submodule(&xmatch_rs_hats)?;

    Ok(())
}
