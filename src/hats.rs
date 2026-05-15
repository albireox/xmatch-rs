/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-05-12
 *  @Filename: hats.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use std::{
    fs::{File, OpenOptions},
    io::{BufReader, Seek},
    path::PathBuf,
    sync::Mutex,
    thread::available_parallelism,
};

use anyhow::{Result, bail};

use polars::lazy::dsl::file_provider as pl_fp;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Configuration options for [`hats_increase_order`].
pub struct HATSIncreaseOrderOptions {
    /// Maximum number of rows allowed per pixel before increasing order.
    ///
    /// If `None`, no automatic pixel selection based on row count is done.
    pub max_rows_per_pixel: Option<usize>,

    /// Maximum number of kilobytes allowed per pixel before increasing order.
    ///
    /// If `None`, no automatic pixel selection based on size is done.
    pub max_kilobytes_per_pixel: Option<u64>,

    /// Whether to keep the original `Norder=<order_0>` directories after processing.
    pub keep_initial_order_directories: bool,

    /// Number of worker threads to use.
    ///
    /// If `None`, the system default parallelism (which is typically the number
    /// of CPU cores) divided by two is used.
    pub n_workers: Option<usize>,

    /// Reference order used to calculate the new pixel indices for `order_1`.
    pub ref_order: u8,

    /// A temporary file to write progress information to.
    ///
    /// The file is written in JSON format and includes the list of pixels being processed
    /// and the list of pixels that have been completed. This can be used to monitor the progress
    /// of the operation from the Python side.
    pub progress_file: Option<PathBuf>,
}

impl HATSIncreaseOrderOptions {
    /// Creates a new options object with a custom `max_rows_per_pixel` value.
    ///
    /// Other options are set to their default values.
    pub fn with_max_rows_per_pixel(max_rows_per_pixel: Option<usize>) -> Self {
        Self {
            max_rows_per_pixel: max_rows_per_pixel,
            ..Self::default()
        }
    }

    /// Returns default options.
    ///
    /// Defaults to `max_rows_per_pixel = Some(1_000_000)` and does not keep initial
    /// order directories.
    pub fn default() -> Self {
        Self {
            max_rows_per_pixel: Some(1_000_000),
            max_kilobytes_per_pixel: None,
            keep_initial_order_directories: false,
            n_workers: None,
            ref_order: 29,
            progress_file: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HATSIncreaseOrderProgress {
    pub pixels_to_process: Vec<i64>,
    pub pixels_completed: Vec<i64>,
}

/// Increases HATS order from `order_0` to `order_1` for selected `pixels`.
///
/// If `pixels` is `None`, candidate pixels are selected based on `options.max_rows_per_pixel`.
pub fn hats_increase_order(
    path: PathBuf,
    order_0: u8,
    order_1: u8,
    pixels: Option<Vec<i64>>,
    options: HATSIncreaseOrderOptions,
) -> Result<()> {
    let mut path = path.clone();

    // Check the parent directory.
    if !path.exists() {
        bail!("Path does not exist: {:?}", path);
    }

    // In case we have specified the top-level HATS we check if there is a
    // "dataset" subdirectory and use it if it exists.
    if path.join("dataset").exists() {
        path = path.join("dataset");
    }

    // Check that the Norder=order_0 directory exists.
    let order_0_dir = path.join(format!("Norder={}", order_0));
    if !order_0_dir.exists() {
        bail!(
            "Norder={} directory does not exist: {:?}",
            order_0,
            order_0_dir
        );
    }

    // List of pixels to increase order for. If the input pixels is None, we
    // list the pixels that have more than `options.max_rows_per_pixel` rows.
    let mut pixels = if let Some(pixels) = pixels {
        pixels
    } else {
        list_pixels_to_increase_order(
            &path,
            order_0,
            options.max_rows_per_pixel,
            options.max_kilobytes_per_pixel,
        )?
    };

    // Sort the pixel list.
    pixels.sort();

    if pixels.is_empty() {
        return Ok(());
    }

    // If there is a progress file configured, write the initial progress information
    // with the list of pixels to process.
    let progress_file = if let Some(progress_file) = options.progress_file {
        let progress_file = Arc::new(Mutex::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(progress_file)
                .unwrap(),
        ));

        let progress = HATSIncreaseOrderProgress {
            pixels_to_process: pixels.clone(),
            pixels_completed: vec![],
        };

        serde_json::to_writer(&*progress_file.lock().unwrap(), &progress).unwrap();

        Some(progress_file)
    } else {
        None
    };

    let n_workers = if let Some(n_workers) = options.n_workers {
        n_workers
    } else {
        available_parallelism().unwrap().get() / 2
    };

    if n_workers > 1 {
        // Use a rayon thread pool for parallel processing.
        let pool = super::tools::create_pool(Some(n_workers))?;
        pool.install(|| {
            use rayon::prelude::*;
            pixels.par_iter().for_each(|pixel| {
                let progress_file_clone = progress_file.clone();
                process_pixel(
                    &path,
                    pixel,
                    order_0,
                    order_1,
                    options.ref_order,
                    options.keep_initial_order_directories,
                    progress_file_clone,
                )
                .unwrap();
            });
        });
    } else {
        // Process pixels sequentially.
        pixels.iter().for_each(|pixel| {
            let progress_file_clone = progress_file.clone();
            process_pixel(
                &path,
                pixel,
                order_0,
                order_1,
                options.ref_order,
                options.keep_initial_order_directories,
                progress_file_clone,
            )
            .unwrap();
        });
    }

    Ok(())
}

/// Lists pixel IDs that exceed the configured maximum row count or byte size.
///
/// Returns an empty vector when `max_rows_per_pixel` is `None`.
fn list_pixels_to_increase_order(
    path: &PathBuf,
    order_0: u8,
    max_rows_per_pixel: Option<usize>,
    max_kilobytes_per_pixel: Option<u64>,
) -> Result<Vec<i64>> {
    if max_rows_per_pixel.is_none() && max_kilobytes_per_pixel.is_none() {
        // Nothing to do here.
        return Ok(vec![]);
    }

    let mut pixels = vec![];

    if let Some(max_rows_per_pixel) = max_rows_per_pixel {
        // Scan the parquet files in the initial order dataset.
        let pl_path = PlRefPath::new(path.to_string_lossy().as_ref());
        let lf = LazyFrame::scan_parquet(pl_path, ScanArgsParquet::default())?;

        // Group by pixel and count rows, then filter pixels that exceed the max row count.
        let lf_pixels = lf
            .clone()
            .group_by([col("Npix")])
            .agg([col("Npix").len().alias("count")])
            .filter(col("count").gt(lit(max_rows_per_pixel as i64)))
            .select([col("Npix")])
            .sort(["Npix"], Default::default())
            .collect()?;

        let max_rows_pixels = lf_pixels
            .column("Npix")?
            .i64()?
            .into_iter()
            .flatten()
            .collect::<Vec<i64>>();

        // Extend the pixels list with the pixels that exceed the max row count.
        pixels.extend(max_rows_pixels);
    };

    if let Some(max_kilobytes_per_pixel) = max_kilobytes_per_pixel {
        // For each Norder=order_0/Dir=*/Npix=* directory, check the total size of
        // the parquet files and add the pixel to the list if it exceeds the
        // max kilobytes per pixel.
        let order_0_dir = path.join(format!("Norder={}", order_0));
        for dir_entry in std::fs::read_dir(order_0_dir)? {
            let dir_entry = dir_entry?;
            let dir_path = dir_entry.path();
            if !dir_path.is_dir()
                || !dir_path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("Dir=")
            {
                continue;
            }

            for npix_entry in std::fs::read_dir(dir_path)? {
                let npix_entry = npix_entry?;
                let npix_path = npix_entry.path();
                let npix_path_str = npix_path.file_name().unwrap().to_string_lossy();

                if !npix_path.is_dir() || !npix_path_str.starts_with("Npix=") {
                    continue;
                }

                let pixel_id: i64 = npix_path_str
                    .strip_prefix("Npix=")
                    .unwrap()
                    .parse()
                    .unwrap();

                if pixels.contains(&pixel_id) {
                    // This pixel is already in the list based on row count, skip it.
                    continue;
                }

                let pixel_size_kb = std::fs::read_dir(&npix_path)?
                    .filter_map(|entry| entry.ok())
                    .filter_map(|entry| entry.metadata().ok())
                    .map(|metadata| metadata.len())
                    .sum::<u64>()
                    / 1024;

                if pixel_size_kb > max_kilobytes_per_pixel {
                    pixels.push(pixel_id);
                }
            }
        }
    };

    // Remove duplicates from the pixels list.
    pixels.sort();
    pixels.dedup();

    Ok(pixels)
}

/// Processes a single pixel and writes re-partitioned parquet output for `order_1`.
fn process_pixel(
    dataset_path: &PathBuf,
    pixel: &i64,
    order_0: u8,
    order_1: u8,
    ref_order: u8,
    keep_initial_order_directories: bool,
    progress_file: Option<Arc<Mutex<File>>>,
) -> Result<()> {
    if order_1 <= order_0 {
        bail!("order_1 must be greater than order_0");
    }

    // Get the directory for the pixel at order_0.
    let dir = pixel / 10_000i64;
    let pixel_dir = dataset_path.join(format!("Norder={}/Dir={}/Npix={}", order_0, dir, pixel));

    if !pixel_dir.exists() {
        bail!("Pixel directory does not exist: {:?}", pixel_dir);
    }

    // Scan the parquet files in the initial order dataset for the given pixel.
    let path_pl = PlRefPath::new(pixel_dir.to_string_lossy().as_ref());
    let lf = LazyFrame::scan_parquet(path_pl, ScanArgsParquet::default())?;

    // Add a column with the new pixel indices at order_1. This is calculated as
    // the integer division of the order_0 pixel index by 4^(ref_order - order_1).
    let k = ref_order - order_1;
    let ref_order_col = format!("_healpix_{}", ref_order);
    let order_1_col = format!("_healpix_{}", order_1);
    let lf_new =
        lf.with_columns([(col(&ref_order_col) / lit(4i64.pow(k as u32))).alias(&order_1_col)]);

    // Sink the new lazyframe to partitioned parquet files by the new pixel index column.
    let dataset_path = dataset_path.join(format!("Norder={}", order_1));
    lf_new
        .sink(
            SinkDestination::Partitioned {
                base_path: PlRefPath::new(dataset_path.to_string_lossy()),
                file_path_provider: crate::tools::file_path_provider_fn!(path_provider_uuid),
                partition_strategy: PartitionStrategy::Keyed {
                    keys: vec![col(&order_1_col)],
                    include_keys: false,
                    keys_pre_grouped: false,
                },
                max_rows_per_file: 1_000_000,
                approximate_bytes_per_file: u64::MAX,
            },
            FileWriteFormat::Parquet(Arc::new(ParquetWriteOptions {
                compression: ParquetCompression::default(),
                statistics: StatisticsOptions::default(),
                row_group_size: None,
                data_page_size: None,
                key_value_metadata: None,
                arrow_schema: None,
                compat_level: None,
            })),
            UnifiedSinkArgs::default(),
        )?
        .collect()?;

    if !keep_initial_order_directories {
        // Remove the original pixel directory for order_0.
        std::fs::remove_dir_all(pixel_dir)?;
    }

    // If there is a progress file configured, update it with the completed pixel.
    if let Some(progress_file) = progress_file {
        // Rewind the progress file to the beginning before reading and writing.
        progress_file.lock().unwrap().rewind().unwrap();

        // Update the progress file with the completed pixel.
        let mut progress: HATSIncreaseOrderProgress =
            serde_json::from_reader(BufReader::new(&*progress_file.lock().unwrap()))?;
        progress.pixels_completed.push(*pixel);

        progress_file.lock().unwrap().rewind().unwrap();
        serde_json::to_writer(&*progress_file.lock().unwrap(), &progress)?;
    }

    Ok(())
}

/// Provides a partitioned output file path for parquet sink writes with a custom
/// UUID and pixel-based directory structure.
fn path_provider_uuid(
    args: pl_fp::FileProviderArgs,
) -> Result<pl_fp::FileProviderReturn, polars::error::PolarsError> {
    // Get the index and partition keys from the arguments. args.partition_keys is a dataframe
    // with one row and (in this case) one column with the pixel index.
    let index = args.index_in_partition;
    let partition_keys = &*args.partition_keys;
    let pixel = partition_keys
        .get(0)
        .unwrap()
        .get(0)
        .unwrap()
        .try_extract::<i64>()
        .unwrap();

    // Calculate the Dir= directory based on the pixel index.
    let dir = pixel / 10_000i64;

    let mut uuid_part = uuid::Uuid::new_v4().to_string();
    uuid_part = uuid_part.replace("-", "");

    let file = format!("{}-{:02}.parquet", uuid_part, index);
    let path = format!("Dir={}/Npix={}/{}", dir, pixel, file);

    return Ok(pl_fp::FileProviderReturn::Path(path));
}
