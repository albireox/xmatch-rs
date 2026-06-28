/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-06-15
 *  @Filename: xmatch.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

#[pyo3::pymodule(name = "xmatch")]
pub(crate) mod py_xmatch {

    use anyhow::Result;
    use log::LevelFilter;
    use pyo3::prelude::*;
    use pyo3_log::{Caching, Logger};
    use pyo3_polars::PyLazyFrame;
    use std::sync::{LazyLock, OnceLock};
    use std::{path::PathBuf, sync::Mutex};

    use crate::xmatch;
    use crate::xmatch::models::{CatalogueModel, XMatchModel, XMatchOptions};
    use crate::xmatch::tools::{move_temp_files, test_progress_tracker};

    // Prevent setting the pyo3_log logger more than once as that fails.
    static LOGGER_SET: LazyLock<Mutex<bool>> = LazyLock::new(|| Mutex::new(false));

    // Global variable to determine if the logger is propagating trace messages.
    static LOGGER_USE_TRACE: OnceLock<bool> = OnceLock::new();

    #[pyfunction(name = "init_logger")]
    pub fn py_init_logger(py: Python, name: String, use_trace: bool) -> Result<()> {
        // The log crate uses :: instead of .
        let target = name.replace(".", "::");

        // Install a new logger if it has not yet been installed.
        // Note that this sets the target but when we call log::info!() and similar, the
        // target is set to the current module. The Python code ensures that the target is
        // always "xmatch_rs" so that any submodule is also captured by the logger.
        let mut is_logger_set = LOGGER_SET.lock().unwrap();
        if !*is_logger_set {
            Logger::new(py, Caching::LoggersAndLevels)?
                .filter_target(
                    target.clone(),
                    if use_trace {
                        LevelFilter::Trace
                    } else {
                        LevelFilter::Debug
                    },
                )
                .install()
                .expect("Someone installed a logger before us.");
            *is_logger_set = true;
        }

        log::set_max_level(if use_trace {
            LevelFilter::Trace
        } else {
            LevelFilter::Debug
        });

        LOGGER_USE_TRACE.set(use_trace).unwrap();

        log::info!(
            "Logger initialized for target '{}', use_trace={}.",
            target,
            use_trace
        );

        Ok(())
    }

    #[pyfunction(name = "get_max_catalogid")]
    pub fn py_get_max_catalogid(xmatch_path: PathBuf, run_id: Option<usize>) -> Result<u64> {
        let max_catalogid = xmatch::tools::get_max_catalogid(Some(xmatch_path), run_id, true)?;
        Ok(max_catalogid)
    }

    #[pyfunction(name = "scan_pixel_from_dataset")]
    pub fn py_scan_pixel_from_dataset(
        hats_path: PathBuf,
        order: u8,
        pixel: u64,
        strict: bool,
    ) -> Option<PyLazyFrame> {
        if let Some(lf) =
            xmatch::tools::scan_pixel_from_hats_dataset(hats_path, order, pixel, strict).unwrap()
        {
            Some(PyLazyFrame(lf))
        } else {
            None
        }
    }

    #[pyfunction(name = "delete_catalogue_from_dataset")]
    pub fn py_delete_catalogue_from_dataset(
        xmatch_path: PathBuf,
        catalogue_name: &str,
        version: &str,
        phase: Option<usize>,
    ) -> Result<()> {
        xmatch::tools::delete_catalogue_from_dataset(xmatch_path, catalogue_name, version, phase)?;

        Ok(())
    }

    #[pyfunction(name = "run_phase_1")]
    pub fn py_run_xmatch_phase_1(
        py: Python<'_>,
        xmatch_path: PathBuf,
        xmatch_model_data: String,
        cat_model_data: String,
        move_files: bool,
        progress_file: Option<PathBuf>,
    ) -> Result<()> {
        let xmatch_data: XMatchModel = serde_json::from_str(&xmatch_model_data)?;
        let cat_data: CatalogueModel = serde_json::from_str(&cat_model_data)?;
        let xmatch_options = XMatchOptions {
            move_files,
            progress_file,
        };

        let phase_1_cl =
            || xmatch::phase_1::run_phase_1(xmatch_path, xmatch_data, cat_data, xmatch_options);

        // We need to detach the Python GIL to allow logging inside the Rayon threads.
        log::trace!("Detaching GIL for phase 1 with trace logging.");
        py.detach(phase_1_cl)
    }

    #[pyfunction(name = "run_phase_2")]
    pub fn py_run_xmatch_phase_2(
        py: Python<'_>,
        xmatch_path: PathBuf,
        xmatch_model_data: String,
        cat_model_data: String,
        move_files: bool,
        progress_file: Option<PathBuf>,
    ) -> Result<()> {
        let xmatch_data: XMatchModel = serde_json::from_str(&xmatch_model_data)?;
        let cat_data: CatalogueModel = serde_json::from_str(&cat_model_data)?;
        let xmatch_options = XMatchOptions {
            move_files,
            progress_file,
        };

        let phase_2_cl =
            || xmatch::phase_2::run_phase_2(xmatch_path, xmatch_data, cat_data, xmatch_options);

        // We need to detach the Python GIL to allow logging inside the Rayon threads.
        log::trace!("Detaching GIL for phase 2 with trace logging.");
        py.detach(phase_2_cl)
    }

    #[pyfunction(name = "run_phase_3")]
    pub fn py_run_xmatch_phase_3(
        py: Python<'_>,
        xmatch_path: PathBuf,
        xmatch_model_data: String,
        cat_model_data: String,
        move_files: bool,
        progress_file: Option<PathBuf>,
    ) -> Result<()> {
        let xmatch_data: XMatchModel = serde_json::from_str(&xmatch_model_data)?;
        let cat_data: CatalogueModel = serde_json::from_str(&cat_model_data)?;
        let xmatch_options = XMatchOptions {
            move_files,
            progress_file,
        };

        let phase_3_cl =
            || xmatch::phase_3::run_phase_3(xmatch_path, xmatch_data, cat_data, xmatch_options);

        // We need to detach the Python GIL to allow logging inside the Rayon threads.
        log::trace!("Detaching GIL for phase 3 with trace logging.");
        py.detach(phase_3_cl)
    }

    #[pyfunction(name = "move_temporary_files")]
    pub fn py_move_temporary_files(
        temp_path: PathBuf,
        final_path: PathBuf,
        suffix: Option<String>,
    ) -> Result<()> {
        move_temp_files(temp_path, final_path, suffix)?;
        Ok(())
    }

    #[pyfunction(name = "test_progress_tracker")]
    pub fn py_test_progress_tracker(
        py: Python<'_>,
        progress_file: PathBuf,
        log_messages: bool,
    ) -> Result<()> {
        py.detach(|| test_progress_tracker(progress_file, log_messages))
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> Result<()> {
        // Add the submodule to the sys.path.
        let py = m.py();
        py.import("sys")?
            .getattr("modules")?
            .set_item("xmatch_rs._xmatch_rs_lib.xmatch", m)?;
        Ok(())
    }
}
