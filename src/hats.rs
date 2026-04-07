/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-03-23
 *  @Filename: hats.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use polars::prelude::*;

pub fn hats_dataset(path: &str) {
    // let lf = LazyFrame::scan_parquet(PlRefPath::from(path), Default::default()).unwrap();
    // let gb = lf
    //     .group_by([col("healpix_4")])
    //     .agg()
    //     .with_new_streaming(true);

    // gb.sink(
    //     SinkDestination::Partitioned {
    //         base_path: PlRefPath::from(
    //             "/mnt/mako-big/gallegoj/xmatch_test_data/gaia_dr3_source_test",
    //         ),
    //         partition_strategy: PartitionStrategy::Keyed {
    //             keys: vec![col("healpix_4")],
    //             include_keys: true,
    //             keys_pre_grouped: false,
    //         },
    //         file_path_provider: None,
    //         max_rows_per_file: 2_00_000,
    //         approximate_bytes_per_file: 1_000_000_000,
    //     },
    //     FileWriteFormat::Parquet(Default::default()),
    //     UnifiedSinkArgs {
    //         mkdir: true,
    //         ..UnifiedSinkArgs::default()
    //     },
    // )
    // .unwrap()
    // .collect()
    // .unwrap();
}
