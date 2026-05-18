/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-05-17
 *  @Filename: rechunk.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use std::path::PathBuf;

use anyhow::{Result, bail};
use polars::prelude::*;

use crate::hats::path_providers::{file_path_provider_fn, path_provider_part};

/// Options for rechunking a HATS dataset.
pub struct HATSRechunkOptions {
    /// Whether the given path is the top-level HATS dataset directory.
    ///
    /// If true, the function will look for "dataset" subdirectory and rechunk the parquet
    /// files in the Norder=*/Dir=*/Npix=* subdirectories. If false, it will rechunk the
    /// parquet files directly in the given path.
    pub is_dataset: bool,
    /// Maximum number of rows per output parquet file.
    ///
    /// Default is 1 million rows per file.
    pub max_rows_per_file: Option<u32>,
    /// Approximate number of bytes per output parquet file.
    ///
    /// Default is no limit (u64::MAX).
    pub approximate_bytes_per_file: Option<u64>,
}

/// Rechunks a HATS dataset.
///
/// This function reads the parquet files in the given path (or in the Norder=*/Dir=*/Npix=*
/// subdirectories if `rechunk_options.is_dataset` is true) and writes new parquet files with
/// maximum number of rows and file size as specified in `rechunk_options`.
pub fn rechunk_hats_dataset(path: PathBuf, rechunk_options: HATSRechunkOptions) -> Result<()> {
    // Check the parent directory.
    if !path.exists() {
        bail!("Path does not exist: {:?}", path);
    }

    // In case we have specified the top-level HATS we check if there is a
    // "dataset" subdirectory and use it if it exists.
    let path = if rechunk_options.is_dataset && path.join("dataset").exists() {
        path.join("dataset")
    } else {
        path
    };

    if rechunk_options.is_dataset {
        // List the Norder= directories in the dataset.
        for entry in glob::glob(&format!("{}/Norder=*/Dir=*/Npix=*", path.display()))? {
            match entry {
                Ok(path) => _rechunk_path(&path, &rechunk_options)?,
                Err(e) => bail!("Error listing pixel directories: {:?}", e),
            }
        }
    } else {
        // Just rechunk the given path.
        _rechunk_path(&path, &rechunk_options)?;
    }

    Ok(())
}

fn _rechunk_path(path: &PathBuf, rechunk_options: &HATSRechunkOptions) -> Result<()> {
    // Rechunks a single path.

    // Rename the original directory to a temporary name with _orig suffix.
    let orig_path = path.with_file_name(format!(
        "{}_orig",
        path.file_name().unwrap().to_string_lossy()
    ));
    std::fs::rename(&path, &orig_path)?;

    // Scan the parquet files in the original directory.
    let path_pl = PlRefPath::new(orig_path.to_string_lossy().as_ref());
    let lf = LazyFrame::scan_parquet(path_pl, ScanArgsParquet::default())?;

    // Sink the lazyframe to new parquet files with the configured rechunking options.
    lf.sink(
        SinkDestination::Partitioned {
            base_path: PlRefPath::new(path.to_string_lossy().as_ref()),
            file_path_provider: file_path_provider_fn!(path_provider_part),
            partition_strategy: PartitionStrategy::FileSize,
            max_rows_per_file: rechunk_options.max_rows_per_file.unwrap_or(1_000_000),
            approximate_bytes_per_file: rechunk_options
                .approximate_bytes_per_file
                .unwrap_or(u64::MAX),
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
    .collect_with_engine(Engine::Streaming)?;

    // Remove the original directory with the _orig suffix.
    std::fs::remove_dir_all(orig_path)?;

    Ok(())
}
