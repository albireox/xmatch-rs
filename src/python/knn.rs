/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-06-10
 *  @Filename: knn.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

#[pyo3::pymodule(name = "knn")]
pub(crate) mod py_knn {

    use anyhow::Result;
    use itertools::Itertools;
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
    use pyo3::prelude::*;

    use crate::knn::{match_sky, match_sky_radius};

    /// Retrieves the k nearest neighbors of a set of spherical coordinates
    /// in another set of spherical coordinates.
    #[pyfunction(name = "match_sky")]
    pub(crate) fn py_match_sky<'py>(
        py: Python<'py>,
        coords1: PyReadonlyArray2<f64>,
        coords2: PyReadonlyArray2<f64>,
        k: usize,
    ) -> PyResult<(
        Vec<Bound<'py, PyArray1<usize>>>,
        Vec<Bound<'py, PyArray1<f64>>>,
    )> {
        let coords1: Vec<(f64, f64)> = coords1
            .as_array()
            .into_iter()
            .tuples()
            .map(|(x, y)| ((*x).to_radians(), (*y).to_radians()))
            .collect();
        let coords2: Vec<(f64, f64)> = coords2
            .as_array()
            .into_iter()
            .tuples()
            .map(|(x, y)| ((*x).to_radians(), (*y).to_radians()))
            .collect();

        let results = match_sky(&coords1, &coords2, k)?;

        let distances = results
            .iter()
            .map(|nn| {
                nn.iter()
                    .map(|(dist, _)| (*dist).to_degrees())
                    .collect::<Vec<f64>>()
                    .into_pyarray(py)
            })
            .collect::<Vec<_>>();

        let indices = results
            .iter()
            .map(|nn| {
                nn.iter()
                    .map(|(_, idx)| *idx)
                    .collect::<Vec<usize>>()
                    .into_pyarray(py)
            })
            .collect::<Vec<_>>();

        Ok((indices, distances))
    }

    /// Retrieves the neighbors of a set of spherical coordinates `coords2` in `coord1` that fall
    /// within a specified radius.
    #[pyfunction(name = "match_sky_radius")]
    pub(crate) fn py_match_sky_radius<'py>(
        py: Python<'py>,
        coords1: PyReadonlyArray2<f64>,
        coords2: PyReadonlyArray2<f64>,
        radius: f64,
    ) -> PyResult<(
        Vec<Bound<'py, PyArray1<usize>>>,
        Vec<Bound<'py, PyArray1<f64>>>,
    )> {
        let coords1: Vec<(f64, f64)> = coords1
            .as_array()
            .into_iter()
            .tuples()
            .map(|(x, y)| ((*x).to_radians(), (*y).to_radians()))
            .collect();
        let coords2: Vec<(f64, f64)> = coords2
            .as_array()
            .into_iter()
            .tuples()
            .map(|(x, y)| ((*x).to_radians(), (*y).to_radians()))
            .collect();

        let results = match_sky_radius(&coords1, &coords2, radius.to_radians())?;

        let distances = results
            .iter()
            .map(|nn| {
                nn.iter()
                    .map(|(dist, _)| (*dist).to_degrees())
                    .collect::<Vec<f64>>()
                    .into_pyarray(py)
            })
            .collect::<Vec<_>>();

        let indices = results
            .iter()
            .map(|nn| {
                nn.iter()
                    .map(|(_, idx)| *idx)
                    .collect::<Vec<usize>>()
                    .into_pyarray(py)
            })
            .collect::<Vec<_>>();

        Ok((indices, distances))
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> Result<()> {
        // Add the submodule to the sys.path.
        let py = m.py();
        py.import("sys")?
            .getattr("modules")?
            .set_item("xmatch_rs._xmatch_rs_lib.knn", m)?;
        Ok(())
    }
}
