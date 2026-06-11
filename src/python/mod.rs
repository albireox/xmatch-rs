/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-05-18
 *  @Filename: mod.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use pyo3::{prelude::*, py_run};

mod hats;
mod healpix;
mod knn;

// Define the Python module
#[pymodule(name = "_xmatch_rs_lib")]
fn xmatch_rs_pymodule(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // xmatch_rs.healpix module
    let xmatch_rs_healpix = PyModule::new(_py, "healpix")?;

    xmatch_rs_healpix.add_function(wrap_pyfunction!(
        healpix::py_ra_dec_to_healpix,
        &xmatch_rs_healpix
    )?)?;
    xmatch_rs_healpix.add_function(wrap_pyfunction!(
        healpix::py_healpix_pixel_area,
        &xmatch_rs_healpix
    )?)?;

    py_run!(
        _py,
        xmatch_rs_healpix,
        "import sys; sys.modules['xmatch_rs._xmatch_rs_lib.healpix'] = xmatch_rs_healpix"
    );

    m.add_submodule(&xmatch_rs_healpix)?;

    // xmatch_rs.hats module
    let xmatch_rs_hats = PyModule::new(_py, "hats")?;

    xmatch_rs_hats.add_function(wrap_pyfunction!(hats::hats_repartition, &xmatch_rs_hats)?)?;
    xmatch_rs_hats.add_function(wrap_pyfunction!(hats::hats_rechunk, &xmatch_rs_hats)?)?;
    xmatch_rs_hats.add_function(wrap_pyfunction!(hats::hats_create_index, &xmatch_rs_hats)?)?;
    xmatch_rs_hats.add_function(wrap_pyfunction!(
        hats::hats_get_neighbours,
        &xmatch_rs_hats
    )?)?;
    xmatch_rs_hats.add_function(wrap_pyfunction!(
        hats::hats_create_margin_cache,
        &xmatch_rs_hats
    )?)?;
    xmatch_rs_hats.add_function(wrap_pyfunction!(hats::py_path_along_edge, &xmatch_rs_hats)?)?;

    py_run!(
        _py,
        xmatch_rs_hats,
        "import sys; sys.modules['xmatch_rs._xmatch_rs_lib.hats'] = xmatch_rs_hats"
    );

    m.add_submodule(&xmatch_rs_hats)?;

    // xmatch_rs.knn module
    let xmatch_rs_knn = PyModule::new(_py, "knn")?;
    xmatch_rs_knn.add_function(wrap_pyfunction!(knn::py_match_sky, &xmatch_rs_knn)?)?;
    xmatch_rs_knn.add_function(wrap_pyfunction!(knn::py_match_sky_radius, &xmatch_rs_knn)?)?;
    m.add_submodule(&xmatch_rs_knn)?;

    Ok(())
}
