/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-05-15
 *  @Filename: tools.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use std::thread::available_parallelism;

use anyhow::Result;

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
