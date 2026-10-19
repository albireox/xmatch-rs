#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-06-09
# @Filename: xmatch.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import contextlib
import json
import logging
import os
import pathlib
import shutil
import tempfile
import threading
import time
from logging.handlers import MemoryHandler
from tempfile import NamedTemporaryFile

from typing import TYPE_CHECKING, Literal

import polars
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from sdsstools import Timer
from sdsstools.logger import SDSSLogger

from xmatch_rs import console
from xmatch_rs._xmatch_rs_lib import xmatch as xmatch_lib
from xmatch_rs.hats.hats import HATS
from xmatch_rs.tools import get_xmatch_rs_logger, lazy_frame_row_count
from xmatch_rs.xmatch.models import CatalogueModel, PhaseModel, XMatchModel
from xmatch_rs.xmatch.schemas import CATALOG_SCHEMA, CATALOG_TO_TABLE_SCHEMA


if TYPE_CHECKING:
    from sdsstools.logger import SDSSLogger


__all__ = ["XMatch"]


# Add trace logging level.
logging.addLevelName(5, "TRACE")
logging.Logger.TRACE = lambda self, message, *args, **kws: self._log(  # type: ignore
    5,
    message,
    *args,
    **kws,
)


class PrefixAdapter(logging.LoggerAdapter):
    """A logging adapter that adds a prefix to log messages."""

    def __init__(self, logger: SDSSLogger, prefix: str | None = None):
        # We don't use super() here because it's easy to just set the attributes
        # and we want the logger to have the right SDSSLogger type.
        self.logger = logger
        self.extra = None
        self.merge_extra = False

        self.prefix = prefix

    def set_catalogue_prefix(self, prefix: str | None):
        """Sets the prefix to the catalogue name."""

        if prefix is None:
            self.prefix = None
            return

        self.prefix = f"[{prefix.upper()}]"

    def process(self, msg, kwargs):
        """Processes the log message by adding the prefix if set."""

        if self.prefix is None or self.prefix == "":
            return msg, kwargs

        return f"{self.prefix.strip()} {msg}", kwargs

    @contextlib.contextmanager
    def without_prefix(self):
        """A context manager to temporarily disable the prefix."""

        old_prefix = self.prefix
        self.prefix = None
        try:
            yield
        finally:
            self.prefix = old_prefix


class XMatch:
    """Represenation of the XMatch product."""

    def __init__(
        self,
        path: str | pathlib.Path,
        version: str | None = None,
        epoch: float | None = None,
        pixelation_order: int | None = None,
        overwrite: bool = False,
        verbose: bool = False,
        quiet: bool = False,
        logfile: pathlib.Path | str | bool | None = None,
    ):
        """Initializes the XMatch product.

        Parameters
        ----------
        path
            The path to the XMatch product.
        version
            The version of the XMatch product. If not provided, it will be read from
            the ``xmatch.yaml`` file.
        epoch
            The epoch of the coordinates in the cross-matched catalogues, in Julian
            years. If not provided, it will be read from the ``xmatch.yaml`` file.
        pixelation_order
            The HEALPix order used to pixelate the unit sphere. If not provided, it
            will be read from the ``xmatch.yaml`` file.
        overwrite
            If `True`, ignores the existing ``xmatch.yaml`` file.
        verbose
            If `True`, enables verbose logging.
        quiet
            If `True`, disables all logging.
        logfile
            The path to the log file. If not provided, logs will be printed
            to the console and saved to ``~/.local/share/xmatch_rs/logs/``.
            If the path is not absolute, it will be relative to the XMatch product
            path.

        """

        self.path = pathlib.Path(path)

        if (self.path / "xmatch.yaml").exists() and not overwrite:
            self.model = XMatchModel.read(self.path / "xmatch.yaml")

            if version is not None and self.model.version != version:
                raise RuntimeError(
                    f"Version mismatch: provided version {version!r} does not "
                    f"match version in xmatch.yaml {self.model.version!r}."
                )

            if (
                pixelation_order is not None
                and self.model.pixelation_order != pixelation_order
            ):
                raise RuntimeError(
                    f"Pixelation order mismatch: provided pixelation order "
                    f"{pixelation_order} does not match pixelation order in "
                    f"xmatch.yaml {self.model.pixelation_order}."
                )

            if epoch is not None and self.model.epoch != epoch:
                raise RuntimeError(
                    f"Epoch mismatch: provided epoch {epoch} does not match epoch "
                    f"in xmatch.yaml {self.model.epoch}."
                )

        else:
            if version is None or pixelation_order is None or epoch is None:
                raise RuntimeError(
                    "Version, pixelation order, and epoch must be provided "
                    "if xmatch.yaml does not exist or overwrite is True."
                )

            self.model = XMatchModel(
                version=version,
                epoch=epoch,
                pixelation_order=pixelation_order,
                catalogues=[],
            )
            self.path.mkdir(parents=True, exist_ok=True)
            self.model.write(self.path / "xmatch.yaml")

        self.version = self.model.version
        self.pixelation_order = self.model.pixelation_order
        self.epoch = self.model.epoch

        self.model._file_path = self.path / "xmatch.yaml"

        # Cache to track the maximum catalogid for a run_id.
        self._max_catalogid: dict[int, int] = {}

        # Start logging.
        # It's important that the target of the logger is xmatch_rs to ensure
        # that the Rust code can also log to it.
        self.log = PrefixAdapter(
            get_xmatch_rs_logger("xmatch_rs", log_level=logging.INFO),
            prefix=None,
        )

        if logfile is True:
            logfile = self.path / "xmatch.log"
        elif logfile is False:
            pass
        elif logfile is not None:
            logfile = pathlib.Path(logfile)
            if not logfile.is_absolute():
                logfile = self.path / logfile

        self.init_logger(
            verbose=verbose,
            quiet=quiet,
            logfile=logfile,
        )

    def init_logger(
        self,
        log: SDSSLogger | None = None,
        logfile: str | pathlib.Path | Literal[False] | None = None,
        quiet: bool = False,
        verbose: bool = False,
        trace: bool = False,
    ):
        """Initializes the logger for the XMatch product.

        Parameters
        ----------
        log
            An optional ``SDSSLogger`` to use for logging. If provided, overrides the
            internal logger.
        logfile
            An optional path to a log file. If provided, logs will be written to this
            file in addition to the logger. Otherwise logs will only be written to
            a temporary file and made available if the cross-matching process fails.
        quiet
            If `True`, suppresses all logging output to the console.
        verbose
            If `True`, sets the console logging level to ``DEBUG``.
        trace
            If `True`, sets the file logging level to ``TRACE``. This is useful for
            debugging, but can cause significant performance overhead.

        """

        if verbose and quiet:
            raise ValueError("Cannot set both verbose and quiet to True.")

        if log is not None:
            # Check that the logger name is "xmatch_rs"
            if not log.name == "xmatch_rs":
                raise ValueError("Logger name must be 'xmatch_rs'.")

            self.log = PrefixAdapter(log)

        # Set the global level to 0 so that trace messages go to the file logger,
        # but keep the stream logger at the level set by verbose/quiet.
        self.log.setLevel(5)
        if verbose:
            self.log.logger.sh.setLevel(logging.DEBUG)
        elif quiet:
            self.log.logger.sh.setLevel(logging.CRITICAL)

        if logfile is None:
            logfile = self._get_logfile(file_name=f"xmatch_{self.version}.log")
        elif isinstance(logfile, (str, pathlib.Path)):
            logfile = pathlib.Path(logfile)
            if not logfile.is_absolute():
                logfile = self.path / logfile
        elif logfile is False:
            pass
        else:
            raise ValueError("logfile must be a string, pathlib.Path, None, or False.")

        # Drop any previous file loggers.
        fh_handlers: list[logging.FileHandler] = []
        for handler in self.log.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                fh_handlers.append(handler)

        for handler in fh_handlers:
            self.log.logger.removeHandler(handler)

        # Start a new file logger.
        if logfile is not False:
            self.log.logger.start_file_logger(
                str(logfile),
                rotating=False,
                log_level=5,
            )
            self.log.debug(f"Logging to file {logfile!s}.")

        # Initialise the logger in Rust.
        xmatch_lib.init_logger(self.log.name, use_trace=trace)

    def add_catalogue(
        self,
        hats_path: str | pathlib.Path,
        name: str | None = None,
        run_id: int | None = None,
        schema_path: str | pathlib.Path | None = None,
        skip_phases: list[int] | None = None,
        join_paths: list[tuple[str, str]] = [],
        query_radius: float = 1.0,
        n_workers: int | None = None,
    ):
        """Adds a new catalogue to the cross-matching process.

        Parameters
        ----------
        hats_path
            The path to the HATS collection.
        name
            The name of the catalogue. If `None`, the name will be inferred from the
            catalogue model.
        run_id
            The run ID of the cross-match for this catalogue. Must be specified
            if this is the first catalogue being cross-matched. Otherwise uses
            the run ID of the last catalogue cross-matched.
        schema_path
            The path to the schema file for the catalogue. If `None`, the schema
            will be inferred from the HATS collection.
        skip_phases
            A list of phases to skip. If `None`, all phases will be executed.
        join_paths
            A list of tuples with the paths to the catalogues to join and the names of
            the columns to join on. Each tuple should be of the form
            ``(this_catalogue_column, other_catalogue.other_column)``.
        query_radius
            The query radius used in phase 2, in arcseconds.
        n_workers
            The number of worker threads to use. If `None`, the default for each
            phase will be used.

        """

        # Check if the catalogue already exists in the model.
        if name in self.model.list_catalogue_names():
            raise ValueError(f"Catalogue {name!r} is already in the cross-match file.")

        phases: list[PhaseModel] = []
        for phase in [1, 2, 3]:
            if skip_phases is not None and phase in skip_phases:
                continue

            phases.append(
                PhaseModel(
                    phase=phase,
                    join_paths=join_paths if phase == 1 else [],
                    query_radius=query_radius if phase == 2 else None,
                    n_workers=n_workers,
                )
            )

        cat_model = self.catalogue_model_from_hats(
            name=name,
            hats_path=pathlib.Path(hats_path),
            run_id=run_id,
            phases=phases,
            schema_path=schema_path,
        )

        self.model.catalogues.append(cat_model)
        self.model.write()

    def catalogue_model_from_hats(
        self,
        hats_path: pathlib.Path,
        name: str | None = None,
        run_id: int | None = None,
        phases: list[PhaseModel] = [],
        schema_path: str | pathlib.Path | None = None,
    ) -> CatalogueModel:
        """Creates a catalogue model from a HATS collection.

        Parameters
        ----------
        hats_path
            The path to the HATS collection.
        name
            The name of the catalogue. If `None`, the short name from the HATS
            schema will be used.
        run_id
            The run ID for the catalogue. If `None`, the run ID of the
            last catalogue in the model will be used.
        phases
            A list of phase models for the catalogue. Defaults to an empty list.
        schema_path
            The path to the schema file for the catalogue. If `None`, the schema
            will be inferred from the HATS collection.

        Returns
        -------
        CatalogueModel
            The created catalogue model.

        """

        hats_path = pathlib.Path(hats_path)

        if run_id is None:
            if len(self.model.catalogues) == 0:
                raise ValueError("run_id must be provided for the first catalogue.")
            run_id = self.model.catalogues[-1].run_id

        try:
            hats = HATS(hats_path, schema=schema_path)
            hats.validate()
        except Exception as err:
            raise RuntimeError(
                f"Error initializing HATS collection {hats_path!s}."
            ) from err

        primary_key = hats.schema.model.primary_key

        name = name or hats.schema.model.short_name
        ra_column = hats.schema.model.ra_column
        dec_column = hats.schema.model.dec_column
        ra_column_unit = hats.schema.model.ra_column_unit
        dec_column_unit = hats.schema.model.dec_column_unit
        pmra_column = hats.schema.model.pmra_column
        pmdec_column = hats.schema.model.pmdec_column
        pmra_is_cosdec = hats.schema.model.pmra_is_cosdec
        epoch = hats.schema.model.epoch
        epoch_column = hats.schema.model.epoch_column
        epoch_format = hats.schema.model.epoch_format

        if ra_column is None or dec_column is None:
            raise ValueError("RA and Dec columns must be specified in the HATS schema.")

        cat_model = CatalogueModel(
            name=name,
            run_id=run_id,
            hats_path=hats_path,
            primary_key=primary_key,
            phases=phases,
            ra_column=ra_column,
            dec_column=dec_column,
            ra_column_unit=ra_column_unit,
            dec_column_unit=dec_column_unit,
            pmra_column=pmra_column,
            pmdec_column=pmdec_column,
            pmra_is_cosdec=pmra_is_cosdec,
            epoch=epoch,
            epoch_column=epoch_column,
            epoch_format=epoch_format,
        )

        return cat_model

    def run(self, delete_temporary: bool = False, skip_checks: bool = False):
        """Runs the cross-matching process for all catalogues in the model.

        Parameters
        ----------
        delete_temporary
            If `True`, deletes any existing temporary files before starting the
            catalogue processing.
        skip_checks
            If `True`, skips the checks for existing entries in the catalog and
            catalog_to_table datasets, and the check for existing temporary files.

        """

        for cat_model in self.model.catalogues:
            if cat_model.complete:
                self.log.debug(f"Skipping complete catalogue {cat_model.name!r}.")
                continue

            self.log.set_catalogue_prefix(None)
            self.log.info(f"Processing catalogue {cat_model.name!r}.")

            self._process_catalogue(
                cat_model,
                delete_temporary=delete_temporary,
                skip_checks=skip_checks,
            )

    def _process_catalogue(
        self,
        cat_model: CatalogueModel,
        delete_temporary: bool = False,
        skip_checks: bool = False,
    ):
        """Processes a catalogue and updates the XMatch model."""

        # Set the prefix for the logger to the catalogue name.
        self.log.set_catalogue_prefix(cat_model.name)

        # Check if all phases are complete. If so, we can skip the
        # processing of this catalogue.
        if all(phase.complete for phase in cat_model.phases):
            self.log.debug(
                f"All phases for catalogue {cat_model.name!r} are complete. "
                "Marking catalogue as cross-matched."
            )
            cat_model.complete = True
            self.model.write()
            return

        if not skip_checks:
            # Check if the xmatch product catalog or catalog_to_table datasets already
            # have entries for this catalogue.
            catalog_lf = self.get_as_lazyframe(table_name=None)
            catalog_lf = catalog_lf.filter(
                polars.col.added_by == cat_model.name,
                polars.col.version == self.version,
            )
            if lazy_frame_row_count(catalog_lf.limit(1)) > 0:
                raise RuntimeError(
                    f"Catalogue {cat_model.name!r} already has entries in the catalog "
                    "dataset. Please, remove them before processing this catalogue."
                )

            catalog_to_table_lf = self.get_as_lazyframe(table_name=cat_model.name)
            catalog_to_table_lf = catalog_to_table_lf.filter(
                polars.col.version == self.version,
            )
            if lazy_frame_row_count(catalog_to_table_lf.limit(1)) > 0:
                if not any([phase.complete for phase in cat_model.phases]):
                    raise RuntimeError(
                        f"Catalogue {cat_model.name!r} already has entries in the "
                        f"catalog_to_{cat_model.name} dataset, but no phases are "
                        "marked as complete. Please, remove the entries in the "
                        f"catalog_to_table dataset before processing this catalogue."
                    )
                else:
                    self.log.warning(
                        f"Catalogue {cat_model.name!r} already has entries in "
                        f"the catalog_to_{cat_model.name} dataset. If this is not "
                        "expected, please remove them before processing this catalogue."
                    )

            # Check the temporary path. If it exists and has files in it, raise an error
            # unless delete_temporary is True.
            tmp_path = self._get_temporary_path(cat_model)
            if tmp_path.exists() and any(tmp_path.iterdir()):
                if delete_temporary:
                    self.log.warning(
                        f"Temporary path {tmp_path!s} not empty. Deleting it."
                    )
                    shutil.rmtree(tmp_path)
                else:
                    raise FileExistsError(
                        f"Temporary path {tmp_path!s} exists and is not empty. Use "
                        "delete_temporary=True to delete it before processing the "
                        "catalogue."
                    )

        # Create the temporary path.
        self._get_temporary_path(cat_model).mkdir(parents=True, exist_ok=True)

        # Run each one of the phases for this catalogue.
        try:
            with Timer() as catalogue_timer:
                for phase in cat_model.phases:
                    if phase.complete:
                        self.log.info(f"Skipping complete phase {phase.phase}.")
                        continue

                    with Timer() as phase_timer:
                        if phase.phase == 1:
                            self._run_phase_1(cat_model)
                        elif phase.phase == 2:
                            self._run_phase_2(cat_model)
                        elif phase.phase == 3:
                            self._run_phase_3(cat_model)

                    self.log.info(
                        f"Phase {phase.phase} completed in "
                        f"{phase_timer.elapsed:,.2f} seconds."
                    )

                    phase.complete = True
                    self.model.write()

                self.log.info(
                    f"Catalogue {cat_model.name!r} processed in "
                    f"{catalogue_timer.elapsed:,.2f} seconds."
                )

        except Exception as err:
            self._raise_with_logfile(err)

        if all(phase.complete for phase in cat_model.phases):
            self.log.info("All phases are complete. Marking catalogue complete.")
            cat_model.complete = True
            self.model.write()

    def _run_phase_1(self, cat_model: CatalogueModel):
        """Runs phase 1 of the cross-matching process for a catalogue."""

        with ProgressTracker(log=self.log, label="Phase 1") as progress:
            xmatch_lib.run_phase_1(
                xmatch_path=str(self.path),
                xmatch_model_data=self.model.model_dump_json(),
                cat_model_data=cat_model.model_dump_json(),
                move_files=True,
                progress_file=str(progress.progress_file),
            )

    def _run_phase_2(self, cat_model: CatalogueModel):
        """Runs phase 2 of the cross-matching process for a catalogue."""

        with ProgressTracker(log=self.log, label="Phase 2") as progress:
            xmatch_lib.run_phase_2(
                xmatch_path=str(self.path),
                xmatch_model_data=self.model.model_dump_json(),
                cat_model_data=cat_model.model_dump_json(),
                move_files=True,
                progress_file=str(progress.progress_file),
            )

    def _run_phase_3(self, cat_model: CatalogueModel):
        """Runs phase 3 of the cross-matching process for a catalogue."""

        with ProgressTracker(log=self.log, label="Phase 3") as progress:
            xmatch_lib.run_phase_3(
                xmatch_path=str(self.path),
                xmatch_model_data=self.model.model_dump_json(),
                cat_model_data=cat_model.model_dump_json(),
                move_files=True,
                progress_file=str(progress.progress_file),
            )

    def _get_temporary_path(self, cat_model: CatalogueModel) -> pathlib.Path:
        """Returns a temporary path for a catalogue."""

        name = cat_model.name
        run_id = cat_model.run_id
        version = self.version

        path = self.path / "tmp" / f"{name}_{version}_{run_id}"

        return path

    def _delete_temporary_path(self, cat_model: CatalogueModel):
        """Deletes the temporary path for a catalogue."""

        path = self._get_temporary_path(cat_model)

        if path.exists():
            shutil.rmtree(path)

        # Delete the tmp directory if it is empty.
        if len(os.listdir(self.path / "tmp")) == 0:
            shutil.rmtree(self.path / "tmp")

    def _get_logfile(self, file_name: str = "xmatch.log") -> pathlib.Path:
        """Returns a path to a temporary log file."""

        path = pathlib.Path("~/.local/share/xmatch_rs/logs").expanduser()
        path.mkdir(parents=True, exist_ok=True)

        return path / file_name

    def _raise_with_logfile(
        self,
        message: str | BaseException,
        exception_class: type[BaseException] = RuntimeError,
    ):
        """Raises an exception ans provides the path to the log file."""

        if isinstance(message, BaseException):
            exception_class = message.__class__
            message = str(message)

        logfile = self.log.logger.log_filename
        if logfile is not None:
            message = message.strip()
            if not message.endswith("."):
                message += "."
            message += f" See log file {logfile!s} for more details."

        # with self.log.without_prefix():
        raise exception_class(message)

    def get_as_lazyframe(self, table_name: str | None = None) -> polars.LazyFrame:
        """Returns a lazy frame with the cross-matched catalogue.

        Parameters
        ----------
        table_name
            The name of the table to return. If `None`, returns the the ``catalog``
            dataset. Otherwise returns the ``catalog_to_{table_name}`` dataset.

        """

        if table_name is None:
            catalog_dataset = self.path / "catalog" / "dataset"
            if not catalog_dataset.exists():
                return polars.LazyFrame(None, schema=CATALOG_SCHEMA)
            return polars.scan_parquet(
                catalog_dataset,
                schema=CATALOG_SCHEMA,
            )

        catalog_to_table_dataset = self.path / f"catalog_to_{table_name}" / "dataset"
        if not catalog_to_table_dataset.exists():
            return polars.LazyFrame(None, schema=CATALOG_TO_TABLE_SCHEMA)

        return polars.scan_parquet(
            catalog_to_table_dataset,
            schema=CATALOG_TO_TABLE_SCHEMA,
            extra_columns="ignore",
        )

    def delete_catalogue(
        self,
        catalogue_name: str,
        phase: int | None = None,
    ):
        """Deletes a catalogue from the cross-match dataset.

        Parameters
        ----------
        catalogue_name
            The name of the catalogue to delete.
        phase
            The phase of the cross-match process. If `None`, deletes the catalogue
            from all phases.

        """
        xmatch_lib.delete_catalogue_from_dataset(
            str(self.path),
            catalogue_name,
            self.version,
            phase,
        )

    def _test_progress_tracker(self, log_messages: bool = True):
        """Tests the progress tracker."""

        progress_file = tempfile.NamedTemporaryFile(delete=False).name
        print("Using test progress file:", progress_file)

        with ProgressTracker(
            progress_file=progress_file,
            show_progress=True,
            log=self.log,
            buffer_log_messages=False,
        ):
            xmatch_lib.test_progress_tracker(
                progress_file=progress_file,
                log_messages=log_messages,
            )


class ProgressTracker:
    """A simple progress tracker for the cross-matching process."""

    def __init__(
        self,
        progress_file: str | pathlib.Path | None = None,
        label: str = "Progress",
        show_progress: bool = True,
        buffer_log_messages: bool = False,
        log: PrefixAdapter | None = None,
    ):
        if progress_file is None:
            progress_file = NamedTemporaryFile(
                delete=False,
                prefix="xmatch_progress_",
                suffix=".json",
            ).name

        self.progress_file = pathlib.Path(progress_file)
        self.label = label
        self.show_progress = show_progress

        self.buffer_log_messages = buffer_log_messages
        self.log = log
        self.memory_handler: MemoryHandler | None = None

        self._monitor_process_task: threading.Thread | None = None
        self._file_process_task: threading.Thread | None = None

        self.pbar: Progress | None = None
        self.pbar_task: TaskID | None = None

    def __enter__(self):
        if not self.show_progress:
            return self

        # Buffer log messages in a MemoryHandler to avoid log messages being printed
        # to the console while the progress bar is active. They still go to the log
        # file. We impose a limit of 10 messages and a flush level of ERROR, so that
        # if an error occurs all the messages are flushed.
        if self.buffer_log_messages and self.log:
            self.memory_handler = MemoryHandler(
                capacity=50,
                flushLevel=logging.ERROR,
                target=self.log.logger.sh,
            )

            # Add the memory handler to the logger and remove the stream handler.
            self.log.logger.addHandler(self.memory_handler)
            self.log.logger.removeHandler(self.log.logger.sh)

        self._monitor_process_task = threading.Thread(
            target=self._monitor_progress,
            daemon=True,
        )
        self._monitor_process_task.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._monitor_process_task is not None:
            self._monitor_process_task.join(timeout=5)

        if self.memory_handler:
            # Flush the memory handler and remove it from the logger.
            self.memory_handler.flush()
            if self.log:
                self.log.logger.addHandler(self.log.logger.sh)
                self.log.logger.removeHandler(self.memory_handler)

        if self.progress_file.exists():
            self.progress_file.unlink()

        if self.pbar is not None and self.pbar_task is not None:
            self.pbar.update(self.pbar_task, visible=False)
            self.pbar.stop()

        self.pbar = None
        self.pbar_task = None

        return False

    def _monitor_file(self) -> None:
        """Checks if the progress bar does not exist and hides the progress bar."""

        if self.pbar and self.progress_file and not self.progress_file.exists():
            self.pbar.stop()
            self.pbar = None
            self.pbar_task = None

            if self._monitor_process_task is not None:
                self._monitor_process_task.join(timeout=5)
                self._monitor_process_task = None

            self.__exit__(None, None, None)

            return

        time.sleep(0.05)

    def _monitor_progress(self) -> None:
        """Monitors the progress and updates the progress bar."""

        self.pbar = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        self.pbar_task: int | None = None

        file_existed = False

        while True:
            if self.progress_file.exists():
                file_existed = True

                self._monitor_file_task = threading.Thread(
                    target=self._monitor_file,
                    daemon=True,
                )

                with open(self.progress_file, "r") as f:
                    try:
                        progress_data = json.load(f)
                    except json.JSONDecodeError:
                        # In case we are reading the file as it's being written
                        # and it's not valid JSON yet, we just wait and try again.
                        time.sleep(1)
                        continue

                    show_progress = progress_data.get("show_progress", True)
                    if not show_progress:
                        break

                    n_total: int = progress_data.get("n_total", 0)
                    n_completed: int = progress_data.get("n_completed", 0)

                if self.pbar_task is None and n_total > 0:
                    self.pbar_task = self.pbar.add_task(
                        self.label,
                        total=n_total,
                    )
                    self.pbar.start()
                elif self.pbar_task is None:
                    time.sleep(1)
                    continue

                self.pbar.update(self.pbar_task, completed=n_completed, total=n_total)
                self.pbar.refresh()

                if n_completed >= n_total:
                    break

            if file_existed and not self.progress_file.exists():
                break

            time.sleep(1)  # Sleep for a bit before checking again

        if self.pbar_task is not None:
            self.pbar.update(self.pbar_task, visible=False)
            self.pbar.stop()
            self.pbar.refresh()

        return
