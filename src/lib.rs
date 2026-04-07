/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-03-18
 *  @Filename: lib.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use pyo3::{prelude::*, py_run};

pub mod hats;
pub mod healpix;
pub mod polars_expressions;

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
fn test_hats_dataset(path: &str) -> PyResult<()> {
    hats::hats_dataset(path);
    Ok(())
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
    xmatch_rs_hats.add_function(wrap_pyfunction!(test_hats_dataset, &xmatch_rs_hats)?)?;
    py_run!(
        _py,
        xmatch_rs_hats,
        "import sys; sys.modules['xmatch_rs._xmatch_rs_lib.hats'] = xmatch_rs_hats"
    );
    m.add_submodule(&xmatch_rs_hats)?;

    Ok(())
}
