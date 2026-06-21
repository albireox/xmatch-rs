/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-06-15
 *  @Filename: phase_2.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use anyhow::{Result, bail};
use std::{path::PathBuf, sync::OnceLock};

use crate::xmatch::models::{CatalogueModel, XMatchModel, XMatchOptions};

static LOG_PREFIX: OnceLock<String> = OnceLock::new();

pub fn run_phase_2(
    xmatch_path: PathBuf,
    xmatch_data: XMatchModel,
    cat_data: CatalogueModel,
    xmatch_options: XMatchOptions,
) -> Result<()> {
    LOG_PREFIX.get_or_init(|| format!("[{}] - Phase 2 -", cat_data.name.to_uppercase()));

    let phase = if let None = cat_data.get_phase(2) {
        bail!(
            "{} Phase 2 is not defined for catalogue model {}",
            LOG_PREFIX.get().unwrap(),
            cat_data.name
        );
    } else {
        cat_data.get_phase(2).unwrap()
    };

    log::info!(
        "{} Running phase 2 for catalogue {}.",
        LOG_PREFIX.get().unwrap(),
        cat_data.name
    );

    bail!("Phase 2 is not yet implemented.");

    Ok(())
}
