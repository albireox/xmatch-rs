/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-05-18
 *  @Filename: healpix.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

#[pyo3::pymodule(name = "healpix")]
pub(crate) mod py_healpix {
    use anyhow::Result;
    use pyo3::prelude::*;

    use crate::healpix;

    /// Converts RA and Dec to HEALPix pixel.
    #[pyfunction]
    pub(crate) fn py_ra_dec_to_healpix(ra: f64, dec: f64, order: u8) -> PyResult<u64> {
        Ok(healpix::ra_dec_to_healpix(ra, dec, order))
    }

    /// Returns the area of a HEALPix pixel at the given order in square degrees or steradians.
    #[pyfunction]
    pub(crate) fn py_healpix_pixel_area(order: u8, degrees: bool) -> PyResult<f64> {
        Ok(healpix::healpix_pixel_area(order, degrees))
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> Result<()> {
        // Add the submodule to the sys.path.
        let py = m.py();
        py.import("sys")?
            .getattr("modules")?
            .set_item("xmatch_rs._xmatch_rs_lib.healpix", m)?;
        Ok(())
    }
}
