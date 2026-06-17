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
#[pyo3::pymodule(name = "_xmatch_rs_lib")]
mod xmatch_rs_pymodule {
    #[pymodule_export]
    #[pymodule_export]
    use super::hats::py_hats;

    #[pymodule_export]
    use super::healpix::py_healpix;

    #[pymodule_export]
    use super::knn::py_knn;
}
