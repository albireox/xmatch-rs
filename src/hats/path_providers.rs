/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-05-17
 *  @Filename: path_providers.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use polars::lazy::dsl::file_provider as pl_fp;

/// Provides a partitioned output file path for parquet sink writes with a custom
/// UUID and pixel-based directory structure.
pub(crate) fn path_provider_uuid_pixel(
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

/// Generates a path with an UUID base file name.
pub(crate) fn path_provider_uuid(
    args: pl_fp::FileProviderArgs,
) -> Result<pl_fp::FileProviderReturn, polars::error::PolarsError> {
    let index = args.index_in_partition;

    let mut uuid_part = uuid::Uuid::new_v4().to_string();
    uuid_part = uuid_part.replace("-", "");

    let file = format!("{}-{:02}.parquet", uuid_part, index);

    return Ok(pl_fp::FileProviderReturn::Path(file));
}

/// File path provider for partitioned sink that generates file paths like
/// "part-000001.parquet", "part-000002.parquet", etc.
pub(crate) fn path_provider_part(
    args: pl_fp::FileProviderArgs,
) -> Result<pl_fp::FileProviderReturn, polars::error::PolarsError> {
    let index = args.index_in_partition;
    let file = format!("part-{:06}.parquet", index + 1);
    return Ok(pl_fp::FileProviderReturn::Path(file));
}

/// Macro to wrap a file path provider.
macro_rules! file_path_provider_fn {
    ($func:path) => {
        Some(
            polars::lazy::dsl::file_provider::FileProviderType::Function(
                polars::lazy::dsl::file_provider::FileProviderFunction::Rust(SpecialEq::new(
                    std::sync::Arc::new($func),
                )),
            ),
        )
    };
}

pub(crate) use file_path_provider_fn;
