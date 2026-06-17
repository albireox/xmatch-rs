#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-06-09
# @Filename: xmatch.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import datetime
import logging
import os
import pathlib
import shutil
from uuid import uuid4

import numpy
import polars
from astropy.time import Time

from sdsstools.logger import SDSSLogger

from xmatch_rs._xmatch_rs_lib import xmatch as xmatch_lib
from xmatch_rs.file_providers import get_hats_file_provider
from xmatch_rs.hats.hats import HATS
from xmatch_rs.schema import CoordUnit, EpochFormat
from xmatch_rs.tools import get_xmatch_rs_logger, lazy_frame_row_count
from xmatch_rs.xmatch.models import CatalogueModel, PhaseModel, XMatchModel
from xmatch_rs.xmatch.schemas import CATALOG_SCHEMA, CATALOG_TO_TABLE_SCHEMA


__all__ = ["XMatch"]


# Reserve the last 11 bits of a catalogid uint64 for the run ID.
RUN_ID_BIT_SHIFT = 64 - 11


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
            If `True`, suppresses all logging output.
        verbose
            If `True`, sets the logging level to ``DEBUG``.

        """

        # if name in [cat.name for cat in self.model.catalogues]:
        #     raise ValueError(f"Catalogue {name!r} already exists in xmatch.yaml.")

        if verbose and quiet:
            raise ValueError("Cannot set both verbose and quiet to True.")

        if log is not None:
            # Check that the logger name is "xmatch_rs"
            if not log.name == "xmatch_rs":
                raise ValueError("Logger name must be 'xmatch_rs'.")

            self.log = log

        if verbose:
            self.log.set_level(logging.DEBUG)
            self.log.sh.setLevel(logging.DEBUG)
        elif quiet:
            self.log.set_level(logging.CRITICAL)
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
        self.log.start_file_logger(str(logfile), rotating=False)
        self.log.debug(f"Logging to file {logfile!s}.")

        # Initialise the logger in Rust.
        xmatch_lib.init_logger(self.log.name)

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

        self._process_catalogue(cat_model, overwrite=overwrite)
        # try:
        #     self._process_catalogue(cat_model, overwrite=overwrite)
        # except Exception as err:
        #     raise
        #     self._raise_with_logfile(err)

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
                self._run_phase_1(cat_model, phase, overwrite=overwrite)
            elif phase.phase == 2:
                self._run_phase_2(cat_model, phase, overwrite=overwrite)
            elif phase.phase == 3:
                self._run_phase_3(cat_model, phase, overwrite=overwrite)

        self.model.catalogues.append(cat_model)
        self.model.write()

    def _run_phase_1(
        self,
        cat_model: CatalogueModel,
        phase: PhaseModel,
        overwrite: bool = False,
    ):
        """Runs phase 1 of the cross-matching process for a catalogue."""

        self.log.info(f"Running phase 1 for catalogue {cat_model.name!r}.")

    def _run_phase_2(
        self,
        cat_model: CatalogueModel,
        phase: PhaseModel,
        overwrite: bool = False,
    ):
        """Runs phase 2 of the cross-matching process for a catalogue."""

        self.log.info(f"Running phase 2 for catalogue {cat_model.name!r}.")

    def _run_phase_3(
        self,
        cat_model: CatalogueModel,
        phase: PhaseModel,
        overwrite: bool = False,
    ):
        """Runs phase 3 of the cross-matching process for a catalogue."""

        self.log.info(f"Running phase 3 for catalogue {cat_model.name!r}.")

        xmatch_lib.run_phase_3(
            xmatch_path=str(self.path),
            version=self.version,
            epoch=self.epoch,
            pixelation_order=self.pixelation_order,
            catalog_data=cat_model.model_dump_json(),
            output_path=str(self._get_temporary_path(cat_model)),
            n_workers=10,
        )

        self.log.info(f"Phase 3 for catalogue {cat_model.name!r} completed.")

        return

        cat_hats = HATS(cat_model.hats_path)
        primary_key = cat_model.primary_key

        lf = cat_hats.get_lazyframe()

        if len(self.model.catalogues) == 0:
            # If this is the first catalogue we are cross-matching things are a bit
            # easiert because we don't need to filter it in any way.
            pass
        else:
            raise NotImplementedError

        # Add catalogid to the catalog dataset.
        self.log.debug("Adding catalogid to the catalog dataset.")
        lf = self._add_catalogid(lf, cat_model)

        # Select the catalog_to_table columns.
        utc_now = datetime.datetime.now(datetime.timezone.utc)
        added_on = polars.lit(
            utc_now,
            dtype=polars.Datetime(
                time_unit="ms",
                time_zone="UTC",
            ),
        )

        self.log.debug("Selecting catalog_to_table columns.")
        lf_catalog_to_table = (
            lf.clone()
            .with_row_index(offset=1, name="pk")
            .select(
                polars.col.pk.cast(polars.UInt64),
                polars.col.catalogid,
                polars.col(primary_key),
                healpix_29=polars.col._healpix_29,
                version=polars.lit(self.version, dtype=polars.String),
                distance=polars.lit(None, dtype=polars.Float64),
                best=polars.lit(True, dtype=polars.Boolean),
                added_by_phase=polars.lit(phase.phase, dtype=polars.Int8),
                added_on=added_on,
            )
        )
        lf_catalog_to_table = self._ensure_schema(
            lf_catalog_to_table,
            CATALOG_TO_TABLE_SCHEMA,
            allow_extra_columns=True,
        )

        # Write the catalog_to_table dataset to a temporary location.
        base_path = (
            self._get_temporary_path(cat_model)
            / f"catalog_to_{cat_model.name}"
            / "dataset"
        )
        self._check_temporary_path(base_path, overwrite=overwrite)

        self.log.info(f"Partitioning catalog_to_table dataset to {base_path!s}.")
        self.partition_frame(
            lf_catalog_to_table,
            base_path=base_path,
            use_uuid=True,
            use_detailed_filename=True,
            catalogue=cat_model,
        )

        # Select the catalog dataset columns and write it to a temporary location.
        self.log.debug("Selecting catalog dataset columns.")
        lf_catalog = lf.clone().select(
            polars.col.catalogid,
            healpix_29=polars.col._healpix_29,
            ra=polars.col(cat_model.ra_column).cast(polars.Float64),
            dec=polars.col(cat_model.dec_column).cast(polars.Float64),
            pmra=self._col_or_none(cat_model, "pmra_column", polars.Float64),
            pmdec=self._col_or_none(cat_model, "pmdec_column", polars.Float64),
            epoch_column=self._col_or_none(cat_model, "epoch_column", polars.Float32),
            added_by=polars.lit(cat_model.name, dtype=polars.String),
            added_on=added_on,
            version=polars.lit(self.version, dtype=polars.String),
        )
        lf_catalog = self._propagate_coordinates(lf_catalog, cat_model)

        # Drop the epoch_column and add the xmatch epoch column.
        lf_catalog = lf_catalog.drop(["epoch_column"]).with_columns(
            epoch=polars.lit(self.epoch, dtype=polars.Float32)
        )

        lf_catalog = self._ensure_schema(lf_catalog, CATALOG_SCHEMA)

        # Write the catalog dataset to a temporary location.
        base_path = self._get_temporary_path(cat_model) / "catalog" / "dataset"
        self._check_temporary_path(base_path, overwrite=overwrite)

        self.log.info(f"Partitioning catalog dataset to {base_path!s}.")
        self.partition_frame(
            lf_catalog,
            base_path=base_path,
            use_uuid=True,
            use_detailed_filename=True,
            catalogue=cat_model,
        )

    def _add_catalogid(
        self,
        lf: polars.LazyFrame,
        cat_model: CatalogueModel,
    ) -> polars.LazyFrame:
        """Adds the catalogid column to the catalog dataset for a catalogue."""

        max_catalogid = self.get_max_catalogid(run_id=cat_model.run_id)

        lf = lf.with_row_index(name="catalogid").with_columns(
            catalogid=(polars.col.catalogid + max_catalogid + 1).cast(polars.UInt64)
        )

        return lf

    def _col_or_none(
        self,
        model: CatalogueModel,
        field: str,
        dtype: type[polars.DataType] = polars.Float64,
    ) -> polars.Expr:
        """Returns a polars expression for a model field or `None`."""

        value = getattr(model, field)
        if value is None:
            return polars.lit(None, dtype=dtype)

        return polars.col(value).cast(dtype)

    def _ensure_schema(
        self,
        lf: polars.LazyFrame,
        schema: polars.Schema,
        allow_extra_columns: bool = False,
    ) -> polars.LazyFrame:
        """Ensures that a lazy frame has the given schema."""

        lf_schema = lf.collect_schema()

        for name, dtype in schema.items():
            if name not in lf_schema:
                lf = lf.with_columns(polars.lit(None, dtype=dtype).alias(name))
            else:
                lf = lf.with_columns(polars.col(name).cast(dtype))

        columns: list[str] = []
        for name in lf.collect_schema().keys():
            if name in schema:
                columns.append(name)
            elif allow_extra_columns:
                columns.append(name)
            else:
                pass

        # Ensure the columns are in the same order as the schema.
        lf = lf.select(columns)

        return lf

    def _propagate_coordinates(
        self,
        lf: polars.LazyFrame,
        cat_model: CatalogueModel,
    ) -> polars.LazyFrame:
        """Propagates the coordinates of a catalogue to the common epoch.

        This expects a lazy frame with the columns ``ra```, `dec`, `pmra`, `pmdec`, and
        ``epoch_column`` are populated, and determines the best way to propagate the
        coordinates. The ``ra`` and ``dec`` columns are transformed. ``ra`` and
        ``dec`` are converted to degrees if necessary. An additional column, ``epoch``,
        is added to the dataset.

        """

        to_epoch = self.epoch

        ra_expr = polars.col.ra
        dec_expr = polars.col.dec

        # Replace all NaN with nulls.
        ra_expr = ra_expr.fill_nan(None)
        dec_expr = dec_expr.fill_nan(None)
        pmra_expr = polars.col.pmra.fill_nan(None)
        pmdec_expr = polars.col.pmdec.fill_nan(None)
        epoch_col_expr = polars.col.epoch_column.fill_nan(None)

        # Convert to degrees if necessary.
        if cat_model.ra_column_unit == CoordUnit.HOURS:
            ra_expr = ra_expr * 15.0
        elif cat_model.ra_column_unit == CoordUnit.RADIANS:
            ra_expr = ra_expr.degrees()

        if cat_model.dec_column_unit == CoordUnit.RADIANS:
            dec_expr = dec_expr.degrees()

        # If the catalogue has a fixed epoch, use that.
        if cat_model.epoch is not None:
            if cat_model.epoch_format == EpochFormat.JD:
                ref_jyear = float(Time(cat_model.epoch, format="jd").jyear)

            elif cat_model.epoch_format == EpochFormat.JULIAN_YEAR:
                ref_jyear = cat_model.epoch
            else:
                raise ValueError(f"Invalid epoch format {cat_model.epoch_format!r}.")

            ref_year = polars.lit(ref_jyear, dtype=polars.Float32)

        elif cat_model.epoch_column is not None:
            if cat_model.epoch_format == EpochFormat.JD:
                raise NotImplementedError(
                    "Epoch format JD in epoch column is not implemented yet."
                )
            elif cat_model.epoch_format == EpochFormat.JULIAN_YEAR:
                ref_jyear = epoch_col_expr.cast(polars.Float32)
            else:
                raise ValueError(f"Invalid epoch format {cat_model.epoch_format!r}.")

        else:
            # There is not enough information to propagate the coordinates since
            # we don't know the reference epoch. We leave the coordinates unchanged.
            self.log.warning("Not enough information to propagate coordinates.")

        # If pmra is not a true angle, convert it to it.
        if not cat_model.pmra_is_cosdec:
            pmra_expr = pmra_expr * dec_expr.radians().cos()

        # Now actually propagate the coordinates to the common epoch.
        diff_years = (to_epoch - ref_year).fill_null(0.0)

        # For RA we need convert the pmra again to a non-great-circle angle and
        # convert from mas to degrees.
        ra_expr = (
            ra_expr
            + (pmra_expr.fill_null(0.0) / dec_expr.radians().cos() * diff_years)
            / 3600.0
            / 1000
        )

        dec_expr = dec_expr + (pmdec_expr.fill_null(0.0) * diff_years) / 3600.0 / 1000

        # Overwrite the ra, dec, and pm columns.
        lf = lf.with_columns(
            ra=ra_expr,
            dec=dec_expr,
            pmra=pmra_expr,
            pmdec=pmdec_expr,
        )

        return lf

    def _check_temporary_path(self, path: pathlib.Path, overwrite: bool = False):
        """Checks if a temporary path exists."""

        if path.exists():
            if overwrite:
                self.log.warning(f"Overwriting temporary path {path!s}.")
                shutil.rmtree(path)
            else:
                raise FileExistsError(f"Temporary path {path!s} already exists.")

    def _get_temporary_lazyframe(self, cat_model: CatalogueModel) -> polars.LazyFrame:
        """Returns a temporary lazy frame for a catalogue."""

        path = self._get_temporary_path(cat_model) / cat_model.name / "dataset"

        if path.exists():
            return polars.scan_parquet(path)

        table_schema = self._get_catalog_to_table_schema(cat_model)
        return polars.LazyFrame(None, schema=table_schema)

    def _get_temporary_path(self, cat_model: CatalogueModel) -> pathlib.Path:
        """Returns a temporary path for a catalogue."""

        name = cat_model.name
        run_id = cat_model.run_id
        version = self.version

        path = self.path / "tmp" / f"{name}_{version}_{run_id}"
        path.mkdir(parents=True, exist_ok=True)

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

    def partition_frame(
        self,
        lf: polars.LazyFrame,
        base_path: pathlib.Path | str,
        use_uuid: bool = True,
        use_detailed_filename: bool = False,
        catalogue: CatalogueModel | None = None,
    ):
        """Partitions a lazy frame by the pixelation order and writes it to disk.

        Parameters
        ----------
        lf
            The lazy frame to partition.
        base_path
            The base path where to write the partitioned files. The files will be
            written as a Parquet hive-style partitioned dataset, with
            partition keys ``Norder=<pixelation_order>``, ``Dir=<dir_n>``, and
            ``Npix=<pixel>``.
        use_uuid
            Whether to use a UUID in the file name to ensure uniqueness.
        use_detailed_filename
            If `True`, the files written to the ``Npix=<pixel>`` partition will
            include the catalogue name, version, and run ID.
        catalogue
            The catalogue for which to write the partitioned files. Required if
            `use_detailed_filename` is `True`.

        """

        if use_detailed_filename:
            if catalogue is None:
                raise ValueError("catalogue must be passed with use_detailed_filename.")
            suffix = f"{catalogue.name}-{self.version}-{catalogue.run_id}"
        else:
            suffix = None

        healpix_column = f"_healpix_{self.pixelation_order}"

        schema = lf.collect_schema()
        if healpix_column not in schema:
            # Number of orders of difference.
            n_diff = 29 - self.pixelation_order
            lf = lf.with_columns(
                (polars.col.healpix_29 / 4**n_diff)
                .cast(polars.UInt64)
                .alias(healpix_column)
            )

        file_provider = get_hats_file_provider(
            order=self.pixelation_order,
            column_name=healpix_column,
            use_uuid=use_uuid,
            suffix=suffix,
        )

        lf.sink_parquet(
            polars.PartitionBy(
                base_path=base_path,
                file_path_provider=file_provider,
                key=[healpix_column],
                include_key=False,
                max_rows_per_file=1_000_000,
            ),
            engine="streaming",
        )

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
        if lazy_frame_row_count(lf) == 0:
            return True

    def get_max_catalogid(
        self,
        run_id: int | None = None,
        use_cache: bool = True,
    ) -> int:
        """Returns the maximum catalogid for a given run ID.

        Parameters
        ----------
        run_id : int | None
            The run ID for which to find the maximum catalogid. If not provided,
            the maximum catalogid across all run IDs will be returned.
        use_cache : bool, optional
            Whether to use the cached value, if available.

        Returns
        -------
        int
            The maximum catalogid for the given run ID.

        """

        if run_id is not None and run_id in self._max_catalogid and use_cache:
            return self._max_catalogid[run_id]

        lf = self.get_lazyframe()

        if lazy_frame_row_count(lf) == 0:
            if run_id is None:
                return 0
            else:
                max_cid = run_id << RUN_ID_BIT_SHIFT
                self._max_catalogid[run_id] = max_cid
                return max_cid

        if run_id is not None:
            # Filter the lazy frame to only include rows with the given run ID.
            max_cid_df = (
                lf.filter(
                    numpy.right_shift(polars.col.catalogid, RUN_ID_BIT_SHIFT) == run_id  # ty: ignore[no-matching-overload]
                )
                .select(polars.col.catalogid)
                .max()
                .collect(engine="streaming")
                .item(0, 0)
            )

            if max_cid_df is None:
                max_cid = run_id << RUN_ID_BIT_SHIFT
            else:
                max_cid = int(max_cid_df)

            self._max_catalogid[run_id] = max_cid
            return max_cid

        max_cid = (
            lf.select(polars.col.catalogid).max().collect(engine="streaming").item(0, 0)
            or 0
        )

        return int(max_cid)

    def _get_catalog_to_table_schema(self, catalog: CatalogueModel):
        """Returns the schema to use for the ``catalog_to_table`` dataset."""

        primary_key = catalog.primary_key

        hats_catalogue = HATS(catalog.hats_path)
        catalogue_pk_column = hats_catalogue.schema.get_column(primary_key)

        if catalogue_pk_column is None:
            raise ValueError(
                f"Primary key column {primary_key!r} not found in HATS catalogue "
                f"{catalog.hats_path!s}."
            )

        dtype = catalogue_pk_column.dtype.to_polars_dtype()

        table_schema = CATALOG_TO_TABLE_SCHEMA.copy()
        table_schema[primary_key] = dtype

        return table_schema
