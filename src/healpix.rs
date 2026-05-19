/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-03-18
 *  @Filename: healpix.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use cdshealpix;
use std::f64::consts::PI;

const SR_PER_SQ_DEG: f64 = (PI / 180.0) * (PI / 180.0);

/// Converts RA and Dec to a HEALPix index at the given order.
pub fn ra_dec_to_healpix(ra: f64, dec: f64, order: u8) -> u64 {
    let nested_layer = cdshealpix::nested::get(order);

    let ra_rad = ra.to_radians();
    let dec_rad = dec.to_radians();

    nested_layer.hash(ra_rad, dec_rad)
}

/// Returns the area of a HEALPix pixel at the given order in square degrees or steradians.
pub fn healpix_pixel_area(order: u8, degrees: bool) -> f64 {
    let nside = 2u64.pow(order as u32);
    let npix = 12 * nside * nside;
    let area_sr = 4.0 * PI / (npix as f64);
    if degrees {
        // One steradian is (180/PI)^2 square degrees.
        area_sr / SR_PER_SQ_DEG
    } else {
        area_sr
    }
}

/// Converts a HEALPix pixel index from order_0 to order_1, returning the new pixel indices.
pub fn healpix_convert_order(pixel: u64, order_0: u8, order_1: u8) -> Vec<u64> {
    if order_1 == order_0 {
        vec![pixel]
    } else if order_1 < order_0 {
        // If we are reducing the order, the pixel in order_1 encompasing the initial
        // order_0 pixel can be calculated by dividing by 4^(order_0 - order_1) (integer division).
        let factor = 4u64.pow((order_0 - order_1) as u32);
        vec![pixel / factor]
    } else {
        // If we are increasing the order, the pixels in order_1 into which the initial order_0
        // pixel is split will be [4^k * p, ..., 4^k * (p+1)−1] where k = order_1 - order_0
        // and p is the pixel index in order_0.
        let factor = 4u64.pow((order_1 - order_0) as u32);
        (0..factor).map(|i| pixel * factor + i).collect()
    }
}
