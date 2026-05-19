/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-05-18
 *  @Filename: bin.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use cdshealpix::nested::get;

fn main() {
    let layer = get(5u8);
    println!("{:?}", layer.vertices(100));
}
