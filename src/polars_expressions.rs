/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-03-18
 *  @Filename: polars_extensions.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use polars::prelude::arity::broadcast_binary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::healpix;

#[derive(Deserialize)]
struct GetHEALPixKwargs {
    depth: u8,
}

/// Polars expression to compute HEALPix index from RA and Dec for a given depth.
#[polars_expr(output_type=UInt64)]
fn get_healpix(inputs: &[Series], kwargs: GetHEALPixKwargs) -> PolarsResult<Series> {
    let ra: &Float64Chunked = inputs[0].f64()?;
    let dec: &Float64Chunked = inputs[1].f64()?;

    let out: UInt64Chunked =
        broadcast_binary_elementwise(ra, dec, |ra: Option<f64>, dec: Option<f64>| {
            match (ra, dec) {
                (Some(ra), Some(dec)) => {
                    Some(healpix::ra_dec_to_healpix(ra, dec, kwargs.depth) as u64)
                }
                _ => None,
            }
        });

    Ok(out.into_series())
}
