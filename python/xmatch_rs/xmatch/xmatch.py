#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-06-09
# @Filename: xmatch.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import logging
import os
import pathlib
import shutil
from uuid import uuid4

import polars

from sdsstools.logger import SDSSLogger

from xmatch_rs._xmatch_rs_lib import xmatch as xmatch_lib
from xmatch_rs.hats.hats import HATS
from xmatch_rs.tools import get_xmatch_rs_logger
from xmatch_rs.xmatch.models import CatalogueModel, PhaseModel, XMatchModel
from xmatch_rs.xmatch.schemas import CATALOG_SCHEMA


__all__ = ["XMatch"]


# Add trace logging level.
logging.addLevelName(5, "TRACE")
logging.Logger.TRACE = lambda self, message, *args, **kws: self._log(  # type: ignore
    5, message, *args, **kws
)


class XMatch:
    """Represenation of the XMatch product."""

    def __init__(
        self,
        path: str | pathlib.Path,
        version: str | None = None,
        epoch: float | None = None,
        pixelation_order: int | None = None,
        overwrite: bool = False,
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

        """

        self.path = pathlib.Path(path)

        # It's important that the target of the logger is xmatch_rs to ensure
        # that the Rust code can also log to it.
        self.log = get_xmatch_rs_logger("xmatch_rs", log_level=logging.WARNING)

        if (self.path / "xmatch.yaml").exists() and not overwrite:
            self.model = XMatchModel.read(self.path / "xmatch.yaml")

            if self.model.version != version:
                self.log.warning(
                    f"Version mismatch: provided version {version!r} does not "
                    f"match version in xmatch.yaml {self.model.version!r}."
                )

            if self.model.pixelation_order != pixelation_order:
                self.log.warning(
                    f"Pixelation order mismatch: provided pixelation order "
                    f"{pixelation_order} does not match pixelation order in "
                    f"xmatch.yaml {self.model.pixelation_order}."
                )

            if self.model.epoch != epoch:
                self.log.warning(
                    f"Epoch mismatch: provided epoch {epoch} does not match epoch "
                    f"in xmatch.yaml {self.model.epoch}."
                )

        else:
            if version is None or pixelation_order is None or epoch is None:
                raise ValueError(
                    "Version, pixelation order, and epoch must be provided "
                    "if xmatch.yaml does not exist or overwrite is True."
                )

            self.model = XMatchModel(
                version=version,
                epoch=epoch,
                pixelation_order=pixelation_order,
                catalogues=[],
            )
            self.model.write(self.path / "xmatch.yaml")

        self.version = self.model.version
        self.pixelation_order = self.model.pixelation_order
        self.epoch = self.model.epoch

        # Cache to track the maximum catalogid for a run_id.
        self._max_catalogid: dict[int, int] = {}

    def get_lazyframe(self, table_name: str | None = None) -> polars.LazyFrame:
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
                return polars.LazyFrame()

        if not catalog_dataset.exists():
            return polars.LazyFrame()

        lf = polars.scan_parquet(catalog_dataset)

        return lf

    def xmatch_catalogue(
        self,
        name: str,
        hats_path: str | pathlib.Path,
        plan: str | None = None,
        run_id: int | None = None,
        skip_phases: list[int] | None = None,
        join_paths: list[tuple[str, str]] = [],
        query_radius: float = 1.0,
        overwrite: bool = False,
        log: SDSSLogger | None = None,
        logfile: str | pathlib.Path | None = None,
        quiet: bool = False,
        verbose: bool = False,
        trace: bool = False,
    ):
        """Cross-matches a new catalogue.

        Parameters
        ----------
        name
            The name of the catalogue.
        hats_path
            The path to the HATS collection.
        plan
            The plan used to cross-match this catalogue.
        run_id
            The run ID of the cross-match for this catalogue. Must be specified
            if this is the first catalogue being cross-matched. Otherwise uses
            the run ID of the last catalogue cross-matched.
        skip_phases
            A list of phases to skip. If `None`, all phases will be executed.
        join_paths
            A list of tuples with the paths to the catalogues to join and the names of
            the columns to join on. Each tuple should be of the form
            ``(this_catalogue_column, other_catalogue.other_column)``.
        query_radius
            The query radius used in phase 2, in arcseconds.
        overwrite
            Overwrites temporary files if they already exist. Use with caution.
        log
            An optional SDSSLogger to use for logging. If provided, overrides the
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

            self.log = log

        # Set the global level to 0 so that trace messages go to the file logger,
        # but keep the stream logger at the level set by verbose/quiet.
        self.log.setLevel(5)
        if verbose:
            self.log.sh.setLevel(logging.DEBUG)
        elif quiet:
            self.log.sh.setLevel(logging.CRITICAL)

        if logfile is None:
            logfile = self._get_logfile(prefix=f"xmatch_{name}_{self.version}")

        # Drop any previous file loggers.
        fh_handlers: list[logging.FileHandler] = []
        for handler in self.log.handlers:
            if isinstance(handler, logging.FileHandler):
                fh_handlers.append(handler)

        for handler in fh_handlers:
            self.log.removeHandler(handler)

        # Start a new file logger.
        self.log.start_file_logger(
            str(logfile),
            rotating=False,
            log_level=5,
        )
        self.log.debug(f"Logging to file {logfile!s}.")

        # Initialise the logger in Rust.
        xmatch_lib.init_logger(self.log.name, use_trace=trace)

        # Check if the catalogue already exists in the model.
        if name in self.model.list_catalogue_names():
            raise ValueError(f"Catalogue {name!r} has already been cross-matched.")

        phases: list[PhaseModel] = []
        for phase in [1, 2, 3]:
            if skip_phases is not None and phase in skip_phases:
                continue

            phases.append(
                PhaseModel(
                    phase=phase,
                    join_paths=join_paths if phase == 1 else [],
                    query_radius=query_radius if phase == 2 else None,
                )
            )

        cat_model = self.catalogue_model_from_hats(
            name=name,
            hats_path=pathlib.Path(hats_path),
            plan=plan,
            run_id=run_id,
            phases=phases,
        )

        # Check the temporary path. If it exists and has files in it, raise an error
        # unless overwrite is True.
        tmp_path = self._get_temporary_path(cat_model)
        if tmp_path.exists() and any(tmp_path.iterdir()):
            if overwrite:
                self.log.warning(f"Overwriting temporary path {tmp_path!s}.")
                shutil.rmtree(tmp_path)
            else:
                raise FileExistsError(f"Temporary path {tmp_path!s} already exists.")

        # Create the temporary path.
        tmp_path.mkdir(parents=True, exist_ok=True)

        is_first = len(self.model.catalogues) == 0
        if self.is_empty():
            if not is_first:
                self._raise_with_logfile(
                    "XMatch directory is empty but there are catalogues in the "
                    "xmatch.yaml file. This may indicate an inconsistency in the "
                    "XMatch product."
                )
            cat_model.delete_phase([1, 2])
        else:
            if is_first:
                self._raise_with_logfile(
                    "XMatch directory is not empty but no catalogues have been "
                    "cross-matched. This may indicate an inconsistency in the "
                    "XMatch product."
                )

        try:
            self._process_catalogue(cat_model, overwrite=overwrite)
        except Exception as err:
            self._raise_with_logfile(err)

    def catalogue_model_from_hats(
        self,
        name: str,
        hats_path: pathlib.Path,
        plan: str | None = None,
        run_id: int | None = None,
        phases: list[PhaseModel] = [],
    ) -> CatalogueModel:

        hats_path = pathlib.Path(hats_path)

        if run_id is None:
            if len(self.model.catalogues) == 0:
                raise ValueError("run_id must be provided for the first catalogue.")
            run_id = self.model.catalogues[-1].run_id

        try:
            hats = HATS(hats_path)
            hats.validate()
        except Exception as err:
            raise RuntimeError(
                f"Error initializing HATS collection {hats_path!s}."
            ) from err

        primary_key = hats.schema.model.primary_key

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
            plan=plan,
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

    def _process_catalogue(self, cat_model: CatalogueModel, overwrite: bool = False):
        """Processes a catalogue and updates the XMatch model."""

        for phase in cat_model.phases:
            if phase.phase == 1:
                self._run_phase_1(cat_model)
            elif phase.phase == 2:
                self._run_phase_2(cat_model)
            elif phase.phase == 3:
                self._run_phase_3(cat_model)

        self.model.catalogues.append(cat_model)
        self.model.write()

    def _run_phase_1(self, cat_model: CatalogueModel):
        """Runs phase 1 of the cross-matching process for a catalogue."""

        xmatch_lib.run_phase_1(
            xmatch_path=str(self.path),
            xmatch_model_data=self.model.model_dump_json(),
            cat_model_data=cat_model.model_dump_json(),
            n_workers=None,
            move_files=True,
        )

    def _run_phase_2(self, cat_model: CatalogueModel):
        """Runs phase 2 of the cross-matching process for a catalogue."""

        self.log.info(f"Running phase 2 for catalogue {cat_model.name!r}.")

    def _run_phase_3(self, cat_model: CatalogueModel):
        """Runs phase 3 of the cross-matching process for a catalogue."""

        xmatch_lib.run_phase_3(
            xmatch_path=str(self.path),
            xmatch_model_data=self.model.model_dump_json(),
            cat_model_data=cat_model.model_dump_json(),
            n_workers=10,
            move_files=True,
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

    def _get_logfile(self, prefix: str = "xmatch") -> pathlib.Path:
        """Returns a path to a temporary log file."""

        path = pathlib.Path("~/.local/share/xmatch_rs/logs").expanduser()
        path.mkdir(parents=True, exist_ok=True)

        uuid = uuid4().hex[:8]
        file_ = f"{prefix}_{uuid}.log"

        return path / file_

    def _raise_with_logfile(
        self,
        message: str | BaseException,
        exception_class: type[BaseException] = RuntimeError,
    ):
        """Raises an exception ans provides the path to the log file."""

        if isinstance(message, BaseException):
            exception_class = message.__class__
            message = str(message)

        logfile = self.log.log_filename
        if logfile is not None:
            message = message.strip()
            if not message.endswith("."):
                message += "."
            message += f" See log file {logfile!s} for more details."

        raise exception_class(message)

    def is_empty(self):
        """Returns `True` if the xmatch directory does not exist or is empty."""

        if not self.path.exists():
            return True

        catalog_dataset = self.path / "catalog" / "dataset"
        if not catalog_dataset.exists():
            return True

        lf = polars.scan_parquet(
            catalog_dataset,
            schema=CATALOG_SCHEMA,
        )
        if (
            lf.select(polars.col.catalogid).limit(1).collect(engine="streaming").height
            == 0
        ):
            return True
