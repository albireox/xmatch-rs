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

use crate::tools;
use crate::xmatch;
use crate::xmatch::models::{JoinPathModel, JoinPathType, XMatchData};

use super::phase_1_column::xmatch_phase1_join_type_column;
use super::phase_1_join_table::xmatch_phase1_join_type_table;

pub(crate) fn run_phase_1(xmatch_data: XMatchData) -> Result<()> {
    let log_prefix = format!(
        "[{}] - Phase 1 -",
        xmatch_data.catalogue_name.to_uppercase()
    );

    let cat_data = xmatch_data
        .get_catalogue_model(&xmatch_data.catalogue_name)
        .ok_or_else(|| {
            anyhow::anyhow!("Catalogue model {} not found", xmatch_data.catalogue_name)
        })?;

    let phase = if let None = cat_data.get_phase(1) {
        bail!(
            "{} Phase 1 is not defined for catalogue model {}",
            log_prefix,
            cat_data.name
        );
    } else {
        cat_data.get_phase(1).unwrap()
    };

    log::info!(
        "{} Running phase 1 for catalogue model {}",
        log_prefix,
        cat_data.name
    );

    // Check join paths.
    let join_paths = &phase.join_paths;
    if join_paths.is_empty() {
        log::info!(
            "{} No join paths defined for phase 1, skipping phase.",
            log_prefix
        );
        return Ok(());
    }

    // Process each join path.
    join_paths.iter().for_each(|join_path| {
        xmatch_phase1_process_join_path(&xmatch_data, join_path).unwrap();
    });

    // Report the number of rows in the catalog_to_table after processing all join paths.
    let tmp_path = xmatch::tools::get_temp_dir_path(
        xmatch_data.path.clone(),
        &cat_data.name,
        &xmatch_data.model.version.clone(),
        cat_data.run_id as usize,
        true,
    )?;

    let catalog_to_table_name = format!("catalog_to_{}", cat_data.name);
    let catalog_to_table_path = xmatch::tools::get_xmatch_path_root(xmatch_data.path.clone())?
        .join(format!("{}/dataset", catalog_to_table_name));

    if catalog_to_table_path.exists() {
        let catalog_to_lf = LazyFrame::scan_parquet(
            tools::to_pl_path(catalog_to_table_path)?,
            Default::default(),
        )?;

        let n_rows = tools::lf_row_count(&catalog_to_lf)?;

        log::info!(
            "{} Finished phase 1 for catalogue {}. catalog_to_{} table has {} rows.",
            log_prefix,
            cat_data.name,
            cat_data.name,
            n_rows.to_formatted_string(&en)
        );
    }

    // Move the original catalogue files if requested.
    if xmatch_data.options.move_files {
        log::info!(
            "{} Moving temporary catalog_to_{} files to xmatch dataset.",
            log_prefix,
            cat_data.name
        );

        let suffix = format!(
            "{}-{}-{}",
            cat_data.name, xmatch_data.model.version, cat_data.run_id
        );

        // catalog_to_table directory
        let catalog_to_table_name = format!("catalog_to_{}", cat_data.name);

        let catalog_to_table_path = xmatch::tools::get_xmatch_path_root(xmatch_data.path.clone())?
            .join(format!("{}/dataset", catalog_to_table_name));
        std::fs::create_dir_all(&catalog_to_table_path)?;

        let tmp_catalog_to_table_path = tmp_path.join(format!("{}/dataset", catalog_to_table_name));

        log::debug!(
            "{} Moving {} dataset from {} to {}.",
            log_prefix,
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
fn xmatch_phase1_process_join_path(
    xmatch_data: &XMatchData,
    join_path_model: &JoinPathModel,
) -> Result<()> {
    let log_prefix = format!(
        "[{}] - Phase 1 -",
        xmatch_data.catalogue_name.to_uppercase()
    );

    // Validate the join path.
    validate_join_path(xmatch_data, join_path_model)?;

    if join_path_model.join_type == JoinPathType::Column {
        xmatch_phase1_join_type_column(xmatch_data, join_path_model)?;
    } else if join_path_model.join_type == JoinPathType::JoinTable {
        xmatch_phase1_join_type_table(xmatch_data, join_path_model)?;
    } else {
        bail!(
            "{} Join path type {:?} not supported for phase 1.",
            log_prefix,
            join_path_model.join_type
        );
    }

    Ok(())
}

fn validate_join_path(xmatch_data: &XMatchData, join_path_model: &JoinPathModel) -> Result<()> {
    let log_prefix = format!(
        "[{}] - Phase 1 -",
        xmatch_data.catalogue_name.to_uppercase()
    );

    let xmatch_model = &xmatch_data.model;

    // Check column names.
    let re_column = Regex::new(r"^([a-zA-Z0-9_]+)$").unwrap();

    let left_on_column = join_path_model.left_on.clone();
    let right_on_column = join_path_model.right_on.clone();

    if let None = re_column.captures(&left_on_column) {
        bail!(
            "{} Invalid left_on column format for phase 1: {}.'",
            log_prefix,
            left_on_column
        );
    }

    if let None = re_column.captures(&right_on_column) {
        bail!(
            "{} Invalid right_on column format for phase 1: {}.'",
            log_prefix,
            right_on_column
        );
    }

    // Confirm that the left table exists in the cross-match.
    let left_table = join_path_model.left_catalogue.clone();
    if xmatch_model.get_catalogue(left_table.as_str()).is_none() {
        bail!(
            "{} Catalogue {} has not been previously cross-matched.",
            log_prefix,
            left_table
        );
    };

    // If we are using a join table, do additional checks.

    if join_path_model.join_type == JoinPathType::JoinTable {
        if join_path_model.join_table_path.is_none() {
            bail!(
                "{} Join path of type 'join_table' does not have a join_table_path.",
                log_prefix,
            );
        };

        if join_path_model.join_table_left_on.is_none() {
            bail!(
                "{} Join path of type 'join_table' does not have a join_table_left_on.",
                log_prefix,
            );
        };

        if join_path_model.join_table_right_on.is_none() {
            bail!(
                "{} Join path of type 'join_table' does not have a join_table_right_on.",
                log_prefix,
            );
        };
    }

    Ok(())
}
