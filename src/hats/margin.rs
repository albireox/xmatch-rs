/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-05-18
 *  @Filename: margin.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use std::fs::{OpenOptions, create_dir_all};
use std::io::{BufReader, Seek};
use std::sync::Mutex;
use std::{f64::consts::PI, path::PathBuf};

use anyhow::{Result, bail};
use cdshealpix::compass_point::Cardinal;
use cdshealpix::nested::bmoc::BMOC;
use cdshealpix::nested::bmoc::Status;
use cdshealpix::nested::{center, get};
use polars::prelude::*;
use polars_rows_iter::df_rows_iter;
use serde::{Deserialize, Serialize};

use crate::hats::path_providers::{file_path_provider_fn, path_provider_part};
use crate::hats::tools as hats_tools;
use crate::healpix::healpix_pixel_area;

/// Options for creating a margin cache.
pub struct MarginCacheOptions {
    /// Number of workers. If `None`, it will use the number of available CPU cores.
    pub n_workers: Option<usize>,
    /// Overwrite the margin cache directory if it already exists.
    pub overwrite: bool,
    /// Progress file path. If `Some`, it will write progress information to this file as JSON.
    pub progress_file: Option<PathBuf>,
}

impl Default for MarginCacheOptions {
    /// Returns the default options for creating a margin cache.
    fn default() -> Self {
        Self {
            n_workers: None,
            overwrite: false,
            progress_file: None,
        }
    }
}

/// Progress information for the margin cache creation.
#[derive(Debug, Serialize, Deserialize)]
pub struct HATSRepartitionProgress {
    pub n_pixels_to_process: usize,
    pub n_pixels_completed: usize,
    pub complete: bool,
}

/// Creates a margin cache inside a collection.
///
/// The collection must be valid and contain the `object` catalogue.
///
/// # Arguments
/// * `collection_path` - The path to the collection.
/// * `margin_name` - The name of the margin cache to create. This will be the
///     name of the directory inside the collection.
/// * `buffer` - The buffer value to use for the margin cache, in arcseconds.
/// * `options` - Options for creating the margin cache.
///
pub fn create_margin_cache(
    collection_path: &PathBuf,
    margin_name: &str,
    buffer: f32,
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
    if margin_cache_path.exists() {
        if options.overwrite {
            std::fs::remove_dir_all(&margin_cache_path)?;
        } else {
            bail!(format!(
                "The margin cache directory {} already exists.",
                margin_cache_path.display()
            ));
        }
    }
    create_dir_all(&margin_cache_path)?;

    // Gets the object dataset and the partition information.
    let object_dataset = hats_tools::get_dataset_directory(&collection_path.join("object"))?;
    let partition_info = hats_tools::PartitionInfo::from_dataset(&object_dataset, true)?;

    // Create an iterator over the rows of the partition (i.e., the pixels to process).
    let pixels = partition_info.get();
    let pixels_iter = df_rows_iter!(&pixels, "Norder" => u8, "Npix" => u64)?;

    // If there is a progress file configured, write the initial progress information
    // with the list of pixels to process.
    let progress_file = if let Some(progress_file) = options.progress_file {
        let progress_file = Arc::new(Mutex::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(progress_file)
                .unwrap(),
        ));

        let progress = HATSRepartitionProgress {
            n_pixels_to_process: pixels.height(),
            n_pixels_completed: 0,
            complete: pixels.height() == 0,
        };

        serde_json::to_writer(&*progress_file.lock().unwrap(), &progress).unwrap();

        Some(progress_file)
    } else {
        None
    };
    // Determine the number of workers to use.
    let n_workers = options.n_workers.unwrap_or(1);

    // Run the margin cache creation in parallel for each pixel.
    let pool = crate::tools::create_pool(Some(n_workers))?;
    pool.install(|| {
        use rayon::prelude::*;

        pixels_iter.par_bridge().for_each(|row| {
            let (norder, npix) = row.unwrap();
            let progress_file_clone = progress_file.clone();

            create_margin_cache_pixel(
                object_dataset.clone(),
                margin_cache_path.clone(),
                partition_info.clone(),
                norder,
                npix,
                buffer,
                progress_file_clone,
            )
            .unwrap();
        });
    });

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
/// * `points` - The strategy to determine the number of points to calculate along the edge.
/// * `wrap` - Whether to wrap the points to ensure they are within the valid range of RA and Dec.
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
    const ARCMIN_TO_RAD: f64 = PI / (180.0 * 60.0);
    const ARCSEC_TO_RAD: f64 = PI / (180.0 * 3600.0);

    let points = match points {
        PointsStrategy::Dynamic => {
            // We approximate the length of the side as the square root of the are of the pixel,
            // which is correct for thr equator. Multiply by 2 to be safe in polar regions.
            // Then ensure we have one point for each 1 arcmin along the path.
            let pixel_area = 4.0 * PI / (12.0 * 2f64.powi(2 * norder as i32));
            let side_length = pixel_area.sqrt();

            let n_points = (2.0 * side_length / ARCMIN_TO_RAD).ceil() as u32;
            if n_points < 100 { 100 } else { n_points }
        }
        PointsStrategy::Fixed(n) => n,
    };

    // Calculate grow amount.
    let do_grow = grow.is_some();
    let grow = grow.unwrap_or(0.0) as f64;
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

    // Direction (x, y) to grow the path for each side.
    let dd = match side {
        CellSide::NW => (-1.0, 1.0),
        CellSide::NE => (1.0, 1.0),
        CellSide::SE => (1.0, -1.0),
        CellSide::SW => (-1.0, -1.0),
    };

    // Get the points along the edge of the cell.
    let side_points = layer.path_along_cell_side(npix, &card1, &card2, true, points);

    // Modify the points to grow the path if needed.
    if do_grow {
        let mut grow_points: Vec<(f64, f64)> = vec![];

        // We want the grown path to reach to the original vertices so that when we stitch
        // all the grown sides to create a polygon the result is correct. For now we just add
        // an additional first and last point which are grown in the direction of the side.
        // This is probably not correct and a better solution would be to interpolate the point
        // in which the sides intercept, but for now it may be good enough.

        let first_point = side_points[0];
        let last_point = side_points[side_points.len() - 1];

        // Add the additional first point.
        const FACTOR: f64 = 1.5;
        grow_points.push(match side {
            CellSide::NW => (
                first_point.0 - FACTOR * grow / f64::cos(first_point.1),
                first_point.1,
            ),
            CellSide::NE => (first_point.0, first_point.1 + FACTOR * grow),
            CellSide::SE => (
                first_point.0 + FACTOR * grow / f64::cos(first_point.1),
                first_point.1,
            ),
            CellSide::SW => (first_point.0, first_point.1 - FACTOR * grow),
        });

        // For each point in the original path, grow it by the specified amount. Note
        // that the amount is grow/sqrt(2) and that we need to take into account the
        // declination when growing in RA.
        for point in &side_points {
            let gpoint_0 = point.0 + (grow / 2.0f64.powf(0.5)) * dd.0 / f64::cos(point.1);
            let gpoint_1 = point.1 + (grow / 2.0f64.powf(0.5)) * dd.1;

            grow_points.push((gpoint_0, gpoint_1));
        }

        // Add the additional last point.
        grow_points.push(match side {
            CellSide::NW => (last_point.0, last_point.1 + FACTOR * grow),
            CellSide::NE => (
                last_point.0 + FACTOR * grow / f64::cos(last_point.1),
                last_point.1,
            ),
            CellSide::SE => (last_point.0, last_point.1 - FACTOR * grow),
            CellSide::SW => (
                last_point.0 - FACTOR * grow / f64::cos(last_point.1),
                last_point.1,
            ),
        });

        // If wrap is true, we need to wrap the points to ensure they are
        // within the valid range of RA and Dec.
        if wrap {
            for gpoint in &mut grow_points {
                if gpoint.0 < 0.0 {
                    gpoint.0 += 2.0 * PI;
                } else if gpoint.0 >= (2.0 * PI) {
                    gpoint.0 -= 2.0 * PI;
                }

                if gpoint.1 < -PI / 2.0 {
                    gpoint.1 += PI;
                } else if gpoint.1 > PI / 2.0 {
                    gpoint.1 -= PI;
                }
            }
        }

        return grow_points.into();
    } else {
        return side_points.into();
    }
}

/// Returns a BMOC covering the margin around a HEALPix cell defined by the specified parameters.
///
/// # Arguments
/// * `norder` - The HEALPix order of the original cell.
/// * `npix` - The HEALPix pixel number.
/// * `buffer` - The buffer value to use for the margin cache, in arcseconds.
/// * `moc_order` - The order of the MOC. If `None`, it will be determined automatically
///   based on the buffer.
///
pub fn get_margin_bmoc(norder: u8, npix: u64, buffer: f32, moc_order: Option<u8>) -> BMOC {
    let mut vertices = Vec::new();

    // Get the polygon vertices along the sides of the cell.
    for side in &[CellSide::NW, CellSide::NE, CellSide::SE, CellSide::SW] {
        let side_points = path_along_edge(
            norder,
            npix,
            *side,
            Some(buffer),
            true,
            PointsStrategy::Dynamic,
            false,
        );
        vertices.extend(side_points);
    }

    // Get the MOC layer. If moc_order is provided, use it. Otherwise, determine
    // it based on the buffer. We iterate over the orders until we find the one that gives a
    // pixel size smaller than the buffer.
    let layer = if let Some(moc_order) = moc_order {
        get(moc_order)
    } else {
        let order = (1u8..=29).find_map(|order| {
            let area = healpix_pixel_area(order, true);
            let res = area.sqrt() * 3600.0; // This is approximate since the pixels are not square, but it is good enough for our purposes.
            if res <= buffer as f64 {
                // Return order - 1. This should speed things up a bit while not adding too
                // many extra targets.
                Some(order - 1)
            } else {
                None
            }
        });

        get(order.unwrap_or(29))
    };

    // Build the BMOC polygon.
    let polygon = layer.polygon_coverage(&vertices, true);

    polygon
}

/// Creates the margin cache for a single pixel.
///
/// # Arguments
/// * `object_dataset` - The path to the object dataset.
/// * `output_path` - The path to the margin cache output directory.
/// * `partition_info` - The partition information for the HATS dataset.
/// * `norder` - The HEALPix order of the pixel.
/// * `npix` - The HEALPix pixel number.
/// * `buffer` - The buffer value to use for the margin cache, in arcseconds.
/// * `progress_file` - An optional progress file to update with the progress
///     of the margin cache creation.
///
fn create_margin_cache_pixel(
    object_dataset: PathBuf,
    output_path: PathBuf,
    partition_info: hats_tools::PartitionInfo,
    norder: u8,
    npix: u64,
    buffer: f32,
    progress_file: Option<Arc<Mutex<std::fs::File>>>,
) -> Result<()> {
    // Get the BMOC covering the margin around the cell.
    let bmoc = get_margin_bmoc(norder, npix, buffer, None);

    // Get the object dataset path.
    let object_dataset_path = hats_tools::get_dataset_directory(&object_dataset)?;

    // Get the neighbours to the cell in the HATS dataset partition.
    let neighbours = hats_tools::get_hats_neighbours(
        object_dataset_path.clone(),
        norder,
        npix,
        Some(partition_info),
    )?;

    // Iterate over the neighbours and collect the targets that fall within the margin.
    let mut frames: Vec<LazyFrame> = vec![];

    for (neigh_order, neigh_npix) in neighbours {
        // Read neighbour pixel data.
        let n_data = hats_tools::read_hats_pixel(&object_dataset_path, neigh_order, neigh_npix)?;

        // We need to clone the BMOC because it is moved inside the closure.
        // BMOC doesn't have clone so we create a new empty BMOC and then use the `or` method
        // to combine it with the original one.
        let bmoc_clone = BMOC::new_empty(12).or(&bmoc);

        // Add a column to the neighbour data indicating whether each target falls within
        // the margin or not. For this we use the _healpix_29 column which we can use to
        // calculate the RA/Dec and then test if it is included in the BMOC.
        let margin_n_data = n_data.with_column(
            col("_healpix_29")
                .map(
                    move |healpix_29_col| {
                        let radec: Vec<bool> = healpix_29_col
                            .u64()
                            .unwrap()
                            .iter()
                            .map(|healpix_29| {
                                let healpix_29 = healpix_29.unwrap();
                                let (ra, dec) = center(29, healpix_29);
                                let status = bmoc_clone.test_coo(ra, dec);
                                match status {
                                    Status::IN => true,
                                    Status::UNKNOWN => true, // Better safe than sorry.
                                    _ => false,
                                }
                            })
                            .collect();

                        Ok(Series::new(PlSmallStr::from("in_margin_bmoc"), radec).into_column())
                    },
                    |_, field| Ok(Field::new(field.name().clone(), DataType::Boolean)),
                )
                .alias("in_margin_bmoc"),
        );

        frames.push(margin_n_data);
    }

    // Concatenate the frames for the neighbours and filter to keep only the targets that fall within the margin.
    let margin_lf = concat(frames, Default::default())?
        .filter(col("in_margin_bmoc"))
        .drop(Selector::ByName {
            names: Arc::new(["in_margin_bmoc".into()]),
            strict: true,
        });

    // Create the output directory for the margin cache if it doesn't exist.
    create_dir_all(&output_path.join("dataset"))?;
    let output_pixel_path = hats_tools::get_pixel_path(&output_path, norder, npix);
    create_dir_all(&output_pixel_path)?;

    // Sink the margin data for the pixel. Use a partition with maximum number of rows per file.
    margin_lf
        .sink(
            SinkDestination::Partitioned {
                base_path: PlRefPath::new(output_pixel_path.to_string_lossy().as_ref()),
                file_path_provider: file_path_provider_fn!(path_provider_part),
                partition_strategy: PartitionStrategy::FileSize,
                max_rows_per_file: 1_000_000,
                approximate_bytes_per_file: u64::MAX,
            },
            FileWriteFormat::Parquet(Default::default()),
            UnifiedSinkArgs::default(),
        )?
        .collect_with_engine(Engine::Streaming)?;

    // Update the progress file if provided.
    if let Some(progress_file) = progress_file {
        let mut p_file = progress_file.lock().unwrap();

        p_file.rewind().unwrap();

        let mut progress: HATSRepartitionProgress =
            serde_json::from_reader(BufReader::new(&*p_file)).unwrap();

        progress.n_pixels_completed += 1;
        progress.complete = progress.n_pixels_completed >= progress.n_pixels_to_process;

        p_file.set_len(0).unwrap();
        p_file.rewind().unwrap();

        serde_json::to_writer(&*p_file, &progress).unwrap();
    }

    Ok(())
}
