/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-05-18
 *  @Filename: margin.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use std::fs::create_dir_all;
use std::{f32::consts::PI, path::PathBuf};

use anyhow::{Result, bail};
use cdshealpix::compass_point::Cardinal;
use cdshealpix::nested::bmoc::BMOC;
use cdshealpix::nested::bmoc::Status;
use cdshealpix::nested::get;
use polars::prelude::*;
use polars_rows_iter::*;
use polygons::{build_search_tree, distances_nearest_edges};

use crate::hats::tools as hats_tools;
use crate::tools as xmatch_tools;

/// Options for creating a margin cache.
pub struct MarginCacheOptions {
    /// Overwrite the margin cache directory if it already exists.
    pub overwrite: bool,
}

impl Default for MarginCacheOptions {
    /// Returns the default options for creating a margin cache.
    fn default() -> Self {
        Self { overwrite: false }
    }
}

/// Creates a margin cache inside a collection.
///
/// The collection must be valid and contain the `object` catalogue.
///
/// # Arguments
/// * `collection_path` - The path to the collection.
/// * `margin_name` - The name of the margin cache to create. This will be the
///     name of the directory inside the collection.
/// * `threshold` - The threshold of the margin cache in arcseconds.
/// * `options` - Options for creating the margin cache.
///
pub fn create_margin_cache(
    collection_path: &PathBuf,
    margin_name: &str,
    threshold: f32,
    options: MarginCacheOptions,
) -> Result<()> {
    // Check if the collection is valid.
    if !hats_tools::is_collection(
        collection_path,
        &hats_tools::IsCollectionProps {
            check_properties: true,
            check_object_catalogue: true,
        },
    ) {
        bail!(format!(
            "The provided path {} is not a valid collection.",
            collection_path.display()
        ));
    }

    // Check path to margin cache.
    let margin_cache_path = collection_path.join(margin_name);
    // if margin_cache_path.exists() {
    //     if options.overwrite {
    //         std::fs::remove_dir_all(&margin_cache_path)?;
    //     } else {
    //         bail!(format!(
    //             "The margin cache directory {} already exists.",
    //             margin_cache_path.display()
    //         ));
    //     }
    // }

    let object_dataset = hats_tools::get_dataset_directory(&collection_path.join("object"))?;
    let partition_info = hats_tools::PartitionInfo::from_dataset(&object_dataset, true)?;

    let pixels = partition_info.get().collect()?;

    let pixels_iter = df_rows_iter!(&pixels, "Norder" => u8, "Npix" => u64)?;

    for row in pixels_iter {
        let (norder, npix) = row?;

        create_margin_cache_pixel(
            object_dataset.clone(),
            margin_cache_path.clone(),
            partition_info.clone(),
            norder,
            npix,
            Some(threshold),
        );
        break;
    }

    Ok(())
}

/// Strategy for determining the number of points to calculate along the edge.
pub enum PointsStrategy {
    /// Estimate the number of points dynamically based on the length of the edge.
    Dynamic,
    /// Use a fixed number of points along each side.
    Fixed(u32),
}

/// The side of a HEALPix cell.
#[derive(Clone, Copy)]
pub enum CellSide {
    NW,
    NE,
    SE,
    SW,
}

/// Returns the path along the edge of a HEALPix cell, optionally growing it.
///
/// # Arguments
/// * `norder` - The HEALPix order of the cell.
/// * `npix` - The HEALPix pixel index of the cell.
/// * `side` - The side of the cell to calculate.
/// * `grow` - The amount to grow the path in arcseconds (if `arcsec` is `true`) or radians.
/// * `arcsec` - Whether the `grow` parameter is in arcseconds (if `true`) or radians.
///
/// # Returns
/// A vector with the points along the edge of the cell. All values are given in radians.
///
pub fn path_along_edge(
    norder: u8,
    npix: u64,
    side: CellSide,
    grow: Option<f32>,
    arcsec: bool,
    points: PointsStrategy,
    wrap: bool,
) -> Vec<(f64, f64)> {
    const ARCSEC_TO_RAD: f32 = PI as f32 / (180.0 * 3600.0);

    let points = match points {
        PointsStrategy::Dynamic => {
            // We approximate the length of the side as the square root of the are of the pixel,
            // which is correct for thr equator. Multiply by 2 to be safe in polar regions.
            // Then ensure we have one point for each 1 arcsec along the path.
            let pixel_area = 4.0 * PI / (12.0 * 2f32.powi(2 * norder as i32));
            let side_length = pixel_area.sqrt();

            (2.0 * side_length / ARCSEC_TO_RAD).ceil() as u32
        }
        PointsStrategy::Fixed(n) => n,
    };

    // Calculate grow amount.
    let do_grow = grow.is_some();
    let grow = grow.unwrap_or(0.0);
    let grow = if arcsec { grow * ARCSEC_TO_RAD } else { grow };

    // Get the layer
    let layer = get(norder);

    // Get the cardinal points and the direction of growth for the specified side.
    let (card1, card2) = match side {
        CellSide::NW => (Cardinal::W, Cardinal::N),
        CellSide::NE => (Cardinal::N, Cardinal::E),
        CellSide::SE => (Cardinal::E, Cardinal::S),
        CellSide::SW => (Cardinal::S, Cardinal::W),
    };

    let dd = match side {
        CellSide::NW => (-1.0, 1.0),
        CellSide::NE => (1.0, 1.0),
        CellSide::SE => (1.0, -1.0),
        CellSide::SW => (-1.0, -1.0),
    };

    // Get the points along the edge of the cell.
    let mut side_points = layer.path_along_cell_side(npix, &card1, &card2, false, points);

    // Modify the points to grow the path if needed.
    if do_grow {
        for point in &mut side_points {
            println!("{}, {}", point.0.to_degrees(), point.1.to_degrees());
            point.0 += (grow as f64 / 2.0f64.powf(0.5) as f64) * dd.0 / f64::cos(point.1);
            point.1 += (grow as f64 / 2.0f64.powf(0.5) as f64) * dd.1;
            println!("---> {}, {}", point.0.to_degrees(), point.1.to_degrees());

            if wrap {
                if point.0 < 0.0 {
                    point.0 += (2.0 * PI) as f64;
                } else if point.0 >= (2.0 * PI) as f64 {
                    point.0 -= (2.0 * PI) as f64;
                }

                if point.1 < 0.0 as f64 {
                    point.1 += (PI / 2.0) as f64;
                } else if point.1 > (PI / 2.0) as f64 {
                    point.1 -= (PI / 2.0) as f64;
                }
            }
        }
    }

    side_points.into()
}

pub fn get_margin_bmoc(norder: u8, npix: u64, grow: Option<f32>, arcsec: bool) -> BMOC {
    let mut vertices = Vec::new();

    for side in &[CellSide::NW, CellSide::NE, CellSide::SE, CellSide::SW] {
        let side_points = path_along_edge(
            norder,
            npix,
            *side,
            grow,
            arcsec,
            PointsStrategy::Fixed(100),
            true,
        );
        vertices.extend(side_points);
    }

    let layer_19 = get(19);
    let polygon = layer_19.polygon_coverage(&vertices, true);

    polygon
}

fn get_polygon_points(norder: u8, npix: u64, grow: Option<f32>) -> Vec<(f64, f64)> {
    let mut vertices = Vec::new();

    for side in &[CellSide::NW, CellSide::NE, CellSide::SE, CellSide::SW] {
        let side_points = path_along_edge(
            norder,
            npix,
            *side,
            grow,
            true,
            PointsStrategy::Fixed(100),
            false,
        );
        vertices.extend(side_points);
    }

    // for point in &mut vertices {
    //     println!(
    //         "Point: RA={}, Dec={}",
    //         point.0.to_degrees(),
    //         point.1.to_degrees()
    //     );
    // }

    vertices
}

pub fn create_margin_cache_pixel(
    object_dataset: PathBuf,
    output_path: PathBuf,
    partition_info: hats_tools::PartitionInfo,
    norder: u8,
    npix: u64,
    threshold: Option<f32>,
) -> Result<()> {
    // let bmoc = get_margin_bmoc(norder, npix, threshold, true);
    let polygon = get_polygon_points(norder, npix, threshold);
    let polygon_tree = build_search_tree(vec![polygon], 4, 4);

    println!("Created BMOC");

    let neighbours = hats_tools::get_hats_neighbours(
        object_dataset.clone(),
        norder,
        npix,
        Some(partition_info.clone()),
    )?;

    let layer_29 = get(29);

    let dataset_path = hats_tools::get_dataset_directory(&object_dataset)?;

    create_dir_all(&output_path.join("dataset"))?;
    let output_pixel_path = hats_tools::get_pixel_path(&output_path, norder, npix);
    create_dir_all(&output_pixel_path)?;

    let mut nn = 1;

    for (neigh_order, neigh_npix) in neighbours {
        println!(
            "Processing neighbour Norder={}, Npix={}",
            neigh_order, neigh_npix
        );
        let data = hats_tools::read_hats_pixel(&dataset_path, neigh_order, neigh_npix)?;

        let polygon_tree_clone = polygon_tree.clone();

        let margin_data = data.with_column(
            col("_healpix_29")
                .map(
                    move |healpix_col| {
                        let radec: Vec<bool> = healpix_col
                            .u64()
                            .unwrap()
                            .iter()
                            .map(|healpix| {
                                let healpix = healpix.unwrap();
                                let (ra, dec) = layer_29.center(healpix);
                                let dist =
                                    distances_nearest_edges(&polygon_tree_clone, &vec![(ra, dec)])
                                        [0];
                                let dist_arcsec = dist * (180.0 * 3600.0) / (PI as f64);
                                dist_arcsec <= threshold.unwrap_or(0.0) as f64
                            })
                            .collect();

                        Ok(Series::new(PlSmallStr::from("test"), radec).into_column())
                    },
                    |_, field| Ok(Field::new(field.name().clone(), DataType::Boolean)),
                )
                .alias("test"),
        );

        let output_pixel_path_neigh = output_pixel_path.join(format!("test-{:04}.parquet", nn));

        margin_data
            .filter(col("test"))
            .sink(
                SinkDestination::File {
                    target: SinkTarget::Path(xmatch_tools::to_pl_path(output_pixel_path_neigh)?),
                },
                FileWriteFormat::Parquet(Default::default()),
                UnifiedSinkArgs::default(),
            )?
            .collect_with_engine(Engine::Streaming)?;

        nn += 1;
    }

    Ok(())
}
