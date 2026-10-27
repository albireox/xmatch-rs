/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-06-15
 *  @Filename: phase_1.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use anyhow::{Result, bail};
use num_format::Locale::en;
use num_format::ToFormattedString;
use polars::prelude::*;
use regex::Regex;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::tools;
use crate::xmatch;
use crate::xmatch::models::JoinPathModel;
use crate::xmatch::models::{CatalogueModel, JoinPathType, XMatchModel, XMatchOptions};

use super::phase_1_column::xmatch_phase1_join_type_column;

static LOG_PREFIX: OnceLock<String> = OnceLock::new();

pub fn run_phase_1(
    xmatch_path: PathBuf,
    xmatch_data: XMatchModel,
    cat_data: CatalogueModel,
    xmatch_options: XMatchOptions,
) -> Result<()> {
    LOG_PREFIX.get_or_init(|| format!("[{}] - Phase 1 -", cat_data.name.to_uppercase()));

    let phase = if let None = cat_data.get_phase(1) {
        bail!(
            "{} Phase 1 is not defined for catalogue model {}",
            LOG_PREFIX.get().unwrap(),
            cat_data.name
        );
    } else {
        cat_data.get_phase(1).unwrap()
    };

    log::info!(
        "{} Running phase 1 for catalogue model {}",
        LOG_PREFIX.get().unwrap(),
        cat_data.name
    );

    // Check join paths.
    let join_paths = &phase.join_paths;
    if join_paths.is_empty() {
        log::info!(
            "{} No join paths defined for phase 1, skipping phase.",
            LOG_PREFIX.get().unwrap()
        );
        return Ok(());
    }

    // Process each join path.
    for join_path in join_paths {
        _xmatch_phase1_process_join_path(
            &xmatch_path,
            &xmatch_data,
            &cat_data,
            &join_path,
            &xmatch_options,
        )?;
    }

    // Report the number of rows in the catalog_to_table after processing all join paths.
    let tmp_path = xmatch::tools::get_temp_dir_path(
        xmatch_path.clone(),
        &cat_data.name,
        &xmatch_data.version.clone(),
        cat_data.run_id as usize,
        true,
    )?;

    let catalog_to_table_name = format!("catalog_to_{}", cat_data.name);
    let catalog_to_table_path = xmatch::tools::get_xmatch_path_root(xmatch_path.clone())?
        .join(format!("{}/dataset", catalog_to_table_name));

    if catalog_to_table_path.exists() {
        let catalog_to_lf = LazyFrame::scan_parquet(
            tools::to_pl_path(catalog_to_table_path)?,
            Default::default(),
        )?;

        let n_rows = tools::lf_row_count(&catalog_to_lf)?;

        log::info!(
            "{} Finished phase 1 for catalogue {}. catalog_to_{} table has {} rows.",
            LOG_PREFIX.get().unwrap(),
            cat_data.name,
            cat_data.name,
            n_rows.to_formatted_string(&en)
        );
    }

    // Move the original catalogue files if requested.
    if xmatch_options.move_files {
        log::info!(
            "{} Moving temporary catalog_to_{} files to xmatch dataset.",
            LOG_PREFIX.get().unwrap(),
            cat_data.name
        );

        let suffix = format!(
            "{}-{}-{}",
            cat_data.name, xmatch_data.version, cat_data.run_id
        );

        // catalog_to_table directory
        let catalog_to_table_name = format!("catalog_to_{}", cat_data.name);

        let catalog_to_table_path = xmatch::tools::get_xmatch_path_root(xmatch_path.clone())?
            .join(format!("{}/dataset", catalog_to_table_name));
        std::fs::create_dir_all(&catalog_to_table_path)?;

        let tmp_catalog_to_table_path = tmp_path.join(format!("{}/dataset", catalog_to_table_name));

        log::debug!(
            "{} Moving {} dataset from {} to {}.",
            LOG_PREFIX.get().unwrap(),
            catalog_to_table_name,
            tmp_catalog_to_table_path.display(),
            catalog_to_table_path.display()
        );

        // Actually move the files.
        xmatch::tools::move_temp_files(
            tmp_catalog_to_table_path.clone(),
            catalog_to_table_path.clone(),
            Some(suffix.clone()),
        )?;

        // Remove the temporary directory if it's empty.
        xmatch::tools::delete_dir_if_empty(tmp_path.parent().unwrap())?;
    }
    Ok(())
}

// Process one join path.
fn _xmatch_phase1_process_join_path(
    xmatch_path: &PathBuf,
    xmatch_data: &XMatchModel,
    cat_data: &CatalogueModel,
    join_path_model: &JoinPathModel,
    xmatch_options: &XMatchOptions,
) -> Result<()> {
    let re_column = Regex::new(r"^([a-zA-Z0-9_]+)$").unwrap();

    let left_on_column = join_path_model.left_on.clone();
    let right_on_column = join_path_model.right_on.clone();

    if let None = re_column.captures(&left_on_column) {
        bail!(
            "{} Invalid left_on column format for phase 1: {}.'",
            LOG_PREFIX.get().unwrap(),
            left_on_column
        );
    }

    if let None = re_column.captures(&right_on_column) {
        bail!(
            "{} Invalid right_on column format for phase 1: {}.'",
            LOG_PREFIX.get().unwrap(),
            right_on_column
        );
    }

    if join_path_model.join_type == JoinPathType::Column {
        xmatch_phase1_join_type_column(
            xmatch_path,
            xmatch_data,
            cat_data,
            join_path_model,
            xmatch_options,
        )?;
    } else {
        bail!(
            "{} Join path type {:?} not supported for phase 1.",
            LOG_PREFIX.get().unwrap(),
            join_path_model.join_type
        );
    }

    Ok(())
}
