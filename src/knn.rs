/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-06-10
 *  @Filename: knn.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use anyhow::Result;
use cdshealpix::haversine_dist;
use kdtree::KdTree;
use rayon::prelude::*;

/// Calculates the spherical distance between two points. Units are in radians.
#[inline]
pub fn spherical_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    let (ra1, dec1) = (a[0], a[1]);
    let (ra2, dec2) = (b[0], b[1]);

    haversine_dist(ra1, dec1, ra2, dec2)
}

/// Matches two sets of spherical coordinates using a k-nearest neighbors search with Haversine distance.
///
/// This function is a simple wrapper arond the [kdtree-rs](https://github.com/mrhooray/kdtree-rs)
/// crate with the Haversine distance as the metric. Tree lookups are parallelised using rayon.
///
/// # Arguments
/// - `coords1`: A vector of tuples containing the right ascension and declination
///     of the first set of coordinates. Units are in radians.
/// - `coords2`: A vector of tuples containing the right ascension and declination
///     of the second set of coordinates. Units are in radians.
/// - `k`: The number of nearest neighbors to search for.
///
/// # Returns
/// A vector with the same length as `coords2`, in which the i-th element is a vector of tuples for
/// all the neighbours of `coords2[i]` in `coords1` up to the k-th neighbour. Each tuple contains
/// the distance to the neighbour and the index of the neighbour in `coords1`.
///
pub fn match_sky(
    coords1: &Vec<(f64, f64)>,
    coords2: &Vec<(f64, f64)>,
    k: usize,
) -> Result<Vec<Vec<(f64, usize)>>> {
    // Initialise the k-d tree.
    let mut kdtree = KdTree::new(2);

    // Build the tree using the first set of coordinates.
    for (i, coord) in coords1.iter().enumerate() {
        kdtree.add([coord.0, coord.1], i)?;
    }

    // For each coordinate in the second set, find the k nearest neighbors in the first set.
    // We use rayon to parallelise the lookups.
    let matches = coords2
        .par_iter()
        .map(|coord| {
            let neighbours = kdtree
                .nearest(&[coord.0, coord.1], k, &spherical_distance)
                .unwrap();

            neighbours
                .into_iter()
                .map(|(dist, idx)| (dist, *idx))
                .collect()
        })
        .collect();

    Ok(matches)
}

/// Matches two sets of spherical coordinates using a radius search with Haversine distance.
///
/// This function is a simple wrapper arond the [kdtree-rs](https://github.com/mrhooray/kdtree-rs)
/// crate with the Haversine distance as the metric. Tree lookups are parallelised using rayon.
///
/// # Arguments
/// - `coords1`: A vector of tuples containing the right ascension and declination
///     of the first set of coordinates. Units are in radians.
/// - `coords2`: A vector of tuples containing the right ascension and declination
///     of the second set of coordinates. Units are in radians.
/// - `radius`: The radius within which to search for neighbors. Units are in radians.
///
/// # Returns
/// A vector with the same length as `coords2`, in which the i-th element is a vector of tuples for
/// all the neighbours of `coords2[i]` in `coords1` within the specified radius. Each tuple contains
/// the distance to the neighbour and the index of the neighbour in `coords1`.
///
pub fn match_sky_radius(
    coords1: &Vec<(f64, f64)>,
    coords2: &Vec<(f64, f64)>,
    radius: f64,
) -> Result<Vec<Vec<(f64, usize)>>> {
    // Initialise the k-d tree.
    let mut kdtree = KdTree::new(2);

    // Build the tree using the first set of coordinates.
    for (i, coord) in coords1.iter().enumerate() {
        kdtree.add([coord.0, coord.1], i)?;
    }

    // For each coordinate in the second set, find the neighbors within the radius in the first set.
    let matches = coords2
        .par_iter()
        .map(|coord| {
            let neighbours = kdtree
                .within(&[coord.0, coord.1], radius, &spherical_distance)
                .unwrap();
            neighbours
                .into_iter()
                .map(|(dist, idx)| (dist, *idx))
                .collect()
        })
        .collect();

    Ok(matches)
}
