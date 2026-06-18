/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-05-15
 *  @Filename: tools.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use std::{path::Path, thread::available_parallelism};

use anyhow::Result;
use polars::{lazy::dsl::file_provider::FileProviderType, prelude::*};

use crate::hats::path_providers::{file_path_provider_fn, path_provider_uuid_pixel};

/// Creates a Rayon thread pool.
///
/// If `num_threads` is `None`, uses the available system parallelism.
pub fn create_pool(num_threads: Option<usize>) -> Result<rayon::ThreadPool> {
    let default_parallelism_approx = available_parallelism().unwrap().get();

    match rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads.unwrap_or(default_parallelism_approx))
        .build()
    {
        Err(e) => Err(e.into()),
        Ok(pool) => Ok(pool),
    }
}

pub fn to_pl_path<T: AsRef<Path>>(path: T) -> Result<PlRefPath> {
    Ok(PlRefPath::from(path.as_ref().to_str().unwrap()))
}

pub trait IsEmpty {
    fn is_empty(&self) -> bool;
}

impl IsEmpty for LazyFrame {
    fn is_empty(&self) -> bool {
        self.clone().limit(1).collect().unwrap().height() == 0
    }
}

/// Options for writing a `LazyFrame` as a partitioned Parquet hive.
pub struct PartitionedParquetSinkOptions {
    pub max_rows_per_file: u32,
    pub approximate_bytes_per_file: u64,
    pub file_path_provider: Option<FileProviderType>,
}

impl Default for PartitionedParquetSinkOptions {
    fn default() -> Self {
        Self {
            max_rows_per_file: 1_000_000,
            approximate_bytes_per_file: u64::MAX,
            file_path_provider: file_path_provider_fn!(path_provider_uuid_pixel),
        }
    }
}

/// Sinks a `LazyFrame` to a partitioned Parquet hive with the given options and partition keys.
pub fn sink_lf_to_partitioned_parquet(
    lf: LazyFrame,
    output_path: impl AsRef<Path>,
    keys: Vec<String>,
    options: PartitionedParquetSinkOptions,
) -> Result<()> {
    lf.sink(
        SinkDestination::Partitioned {
            base_path: to_pl_path(output_path)?,
            file_path_provider: options.file_path_provider,
            partition_strategy: PartitionStrategy::Keyed {
                keys: keys.into_iter().map(col).collect(),
                include_keys: false,
                keys_pre_grouped: false,
            },
            max_rows_per_file: options.max_rows_per_file,
            approximate_bytes_per_file: options.approximate_bytes_per_file,
        },
        FileWriteFormat::Parquet(Arc::new(ParquetWriteOptions::default())),
        UnifiedSinkArgs::default(),
    )?
    .collect_with_engine(Engine::Streaming)?;

    Ok(())
}

pub fn lf_row_count(lf: &LazyFrame) -> Result<u32> {
    let n_rows = lf
        .clone()
        .select([len().cast(DataType::UInt32).alias("n_rows")])
        .collect_with_engine(Engine::Streaming)?
        .unwrap_single()
        .column("n_rows")?
        .u32()?
        .get(0)
        .unwrap();

    Ok(n_rows as u32)
}
