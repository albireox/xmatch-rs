#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-03-17
# @Filename: convert.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import glob
import multiprocessing
import pathlib
from functools import partial

from typing import Callable, Literal, Sequence

import numpy
import polars
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from sdsstools.logger import SDSSLogger, get_logger

from xmatch_rs import console
from xmatch_rs.convert.callbacks import callbacks
from xmatch_rs.expressions import get_healpix
from xmatch_rs.schema import Schema
from xmatch_rs.tools import (
    get_polars_dtype_from_string,
    is_glob_pattern,
    lazy_frame_row_count,
)


__all__: list[str] = ["convert_csv_to_parquet"]


CALLBACK_FUNC_T = Callable[[polars.LazyFrame], polars.LazyFrame]
APPLY_CALLBACK_T = CALLBACK_FUNC_T | str


def convert_csv_to_parquet(
    csv_files: list[str | pathlib.Path] | str | pathlib.Path,
    output_dir: str | pathlib.Path,
    schema: Schema | str | pathlib.Path | dict,
    allow_extra_columns: bool = False,
    mode: Literal["direct", "partition"] = "direct",
    partition_key: str | list[str] | None = None,
    partition_max_rows_per_file: int | None = None,
    partition_approximate_bytes_per_file: int | Literal["auto"] | None = 1000_000_000,
    add_healpix_columns: bool = False,
    healpix_depths: int | Sequence[int] = [18],
    healpix_column_name: str = "healpix_{depth}",
    ra_column: str = "ra",
    dec_column: str = "dec",
    ra_column_unit: Literal["degrees", "hours", "radians"] = "degrees",
    dec_column_unit: Literal["degrees", "radians"] = "degrees",
    add_row_index_column: bool = False,
    row_index_column_name: str = "id",
    apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None = None,
    parallelise: bool = False,
    n_workers: int | None = None,
    scan_csv_kwargs: dict | None = None,
    sink_parquet_kwargs: dict | None = None,
    show_progress: bool = False,
    log: SDSSLogger | None = None,
) -> None:
    """Converts CSV files to Parquet format using the provided schema.

    Parameters
    ----------
    csv_files
        A sequence of paths to the CSV files to convert to Parquet. Can also be
        a single path to a CSV file or a directory containing CSV files, in which
        case all CSV files in the directory will be converted. Glob patters are
        supported.
    output_dir
        The directory where the converted Parquet files will be saved.
    schema
        The schema to use for the conversion. It can be a path to a schema file,
        the name of a schema in the ``schemas/`` directory, or a dictionary
        defining the schema.
    allow_extra_columns
        Whether to allow extra columns in the CSV files that are not defined in the
        schema. The data type of the extra columns will be inferred by Polars.
    mode
        The conversion mode. If ``"direct``, each CSV files will be converted
        to a Parquet files. If ``"partition"``, the CSV files will be partitioned
        according to the provided parameters.
    partition_key
        The key(s) to use for partitioning.
    partition_max_rows_per_file
        The maximum number of rows per partitioned file.
    partition_approximate_bytes_per_file
        The approximate number of bytes per partitioned file. Defaults to 1 GB.
    add_healpix_columns
        Whether to add a HEALPix column to the output Parquet files. If ``True``,
        the HEALPix column will be added using the provided RA and Dec columns and
        the specified depth.
    healpix_depths
        A sequence of depths to use for the HEALPix column(s).
    healpix_column_name
        The name to use for the HEALPix column. Can include ``{depth}``, which
        will be replaced by the actual depth used (this is required if multiple
        depths are provided).
    ra_column
        The name of the column containing the RA values.
    dec_column
        The name of the column containing the Dec values.
    ra_column_unit
        The unit of the RA column. Can be ``"degrees"``, ``"hours"``, or ``"radians"``.
    dec_column_unit
        The unit of the Dec column. Can be ``"degrees"`` or ``"radians"``.
    add_row_index_column
        Whether to add a row index column to the output Parquet files. The columns
        is simply a numerical sequential index starting from 1. This option cannot
        be used if ``mode="direct"`` and ``parallelise=True``.
    row_index_column_name
        The name to use for the row index column.
    apply_callback
        An optional callback function to apply to the lazy frame before writing it to
        Parquet. This can be used to apply additional transformations to the data.
        The function is called after all other transformations (e.g. adding
        HEALPix and row index columns) have been applied. The callback must accept
        a single argument, a :obj:`polars.LazyFrame` object, and return the
        same frame after applying the transformations. The function is called once
        for each CSV file in ``"direct"`` mode, and once for the entire dataset
        in ``"partition"`` mode. Alternatively, the callback function can be
        a string representing one of the predefined callbacks provided by
        ``xmatch_rs``.
    parallelise
        Whether to parallelise the conversion. Only applicable if ``mode`` is
        ``"direct"``.
    n_workers
        The number of workers to use for parallelisation.
    scan_csv_kwargs
        Additional keyword arguments to pass to `polars.scan_csv`.
    sink_parquet_kwargs
        Additional keyword arguments to pass to `polars.LazyFrame.sink_parquet`.
    show_progress
        Whether to show a progress bar during the conversion. Only applicable
        if ``mode`` is ``"direct"``.
    log
        An optional logger to use for logging messages during the conversion.

    """

    log = log or get_logger(
        "xmatch_rs.convert",
        use_rich_handler=True,
        log_level=100,
    )
    log.rich_console = console

    # Retrieve the schema if a path or name is provided.
    if not isinstance(schema, Schema):
        schema = Schema(schema)

    polars_schema = schema.get_polars_schema(
        source_columns=True,
    )

    # Prepare the list of CSV files to convert.
    if isinstance(csv_files, (str, pathlib.Path)):
        csv_files = [csv_files]

    csv_files_tmp: list[pathlib.Path | str] = []
    for file in csv_files:
        if is_glob_pattern(file):
            for matched_file in glob.glob(str(file)):
                csv_files_tmp.append(matched_file)
        else:
            path = pathlib.Path(file)
            if path.is_dir():
                for matched_file in path.glob("*.csv*"):
                    csv_files_tmp.append(matched_file)
            else:
                csv_files_tmp.append(path)
    csv_paths = [pathlib.Path(ff) for ff in csv_files_tmp]

    if len(csv_paths) == 0:
        raise ValueError("No CSV files found to convert.")

    # Prepare the scan and sink kwargs with some defaults.
    scan_csv_kwargs = scan_csv_kwargs or {}
    sink_parquet_kwargs = sink_parquet_kwargs or {}

    if allow_extra_columns:
        scan_csv_kwargs["schema_overrides"] = polars_schema
    else:
        scan_csv_kwargs["schema"] = polars_schema

    source_params = schema.schema.source_catalogue_params
    if source_params is not None:
        if source_params.format != "csv":
            raise ValueError("Schema for catalogue has source_params.format != 'csv'.")

        if source_params.csv_has_header is not None:
            scan_csv_kwargs["has_header"] = source_params.csv_has_header
        if source_params.csv_delimiter is not None:
            scan_csv_kwargs["separator"] = source_params.csv_delimiter
        if source_params.csv_comment_prefix is not None:
            scan_csv_kwargs["comment_prefix"] = source_params.csv_comment_prefix
    else:
        log.warning(
            "No source_params found in schema. Using default "
            "parameters for reading CSV files."
        )

    # Set values for scan_csv_kwargs.
    if "comment_prefix" not in scan_csv_kwargs:
        scan_csv_kwargs["comment_prefix"] = "#"
    if "null_values" not in scan_csv_kwargs:
        scan_csv_kwargs["null_values"] = ["null", "", "NULL"]

    log.info(f"Converting {len(csv_paths)} CSV files to Parquet ...")

    if schema.schema_file:
        log.debug(f"Using schema file: {schema.schema_file!s}")

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "direct":
        pbar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
            expand=True,
            auto_refresh=True,
        )
        pbar_task = pbar.add_task("Converting CSV to Parquet ...", total=len(csv_paths))

        if show_progress:
            pbar.start()

        _imap_partial = partial(
            _convert_csv_to_parquet_direct,
            output_dir=output_dir,
            schema=schema,
            allow_extra_columns=allow_extra_columns,
            scan_csv_kwargs=scan_csv_kwargs,
            sink_parquet_kwargs=sink_parquet_kwargs,
            add_healpix_columns=add_healpix_columns,
            healpix_depths=healpix_depths,
            healpix_column_name=healpix_column_name,
            ra_column=ra_column,
            dec_column=dec_column,
            ra_column_unit=ra_column_unit,
            dec_column_unit=dec_column_unit,
            add_row_index_column=add_row_index_column,
            row_index_column_name=row_index_column_name,
            apply_callback=apply_callback,
        )

        if n_workers == 1:
            parallelise = False

        if parallelise:
            if add_row_index_column:
                raise ValueError(
                    "Cannot add a row index column when parallelising "
                    "the conversion in direct mode."
                )

            proc = multiprocessing.get_context("spawn")
            if n_workers is None:
                n_workers = proc.cpu_count()

            with proc.Pool(n_workers) as pool:
                for _ in pool.imap_unordered(_imap_partial, csv_paths):
                    pbar.advance(pbar_task)

        else:
            row_index_start = 1
            for csv_file in sorted(csv_paths):
                n_rows = _imap_partial(csv_file, row_index_start=row_index_start)
                row_index_start += n_rows
                pbar.advance(pbar_task)

        pbar.stop()

    elif mode == "partition":
        lf = polars.scan_csv([str(ff) for ff in csv_files], **scan_csv_kwargs)

        lf = _process_lazy_frame(
            lf,
            schema,
            ra_column=ra_column,
            dec_column=dec_column,
            ra_column_unit=ra_column_unit,
            dec_column_unit=dec_column_unit,
            add_healpix_columns=add_healpix_columns,
            healpix_depths=healpix_depths,
            healpix_column_name=healpix_column_name,
            add_row_index_column=add_row_index_column,
            row_index_column_name=row_index_column_name,
            row_index_start=1,
            apply_callback=apply_callback,
        )

        partition_by = polars.PartitionBy(
            output_dir,
            key=partition_key,
            include_key=True if partition_key is not None else None,
            max_rows_per_file=partition_max_rows_per_file,
            approximate_bytes_per_file=partition_approximate_bytes_per_file,
        )

        lf.sink_parquet(partition_by, **sink_parquet_kwargs)

    else:
        raise ValueError("Invalid mode. Must be 'direct' or 'partition'.")

    log.info(f"Conversion complete. Parquet files saved to {output_dir!s}.")


def _convert_csv_to_parquet_direct(
    csv_file: pathlib.Path | str,
    output_dir: pathlib.Path,
    schema: Schema,
    allow_extra_columns: bool = False,
    scan_csv_kwargs: dict | None = None,
    sink_parquet_kwargs: dict | None = None,
    add_healpix_columns: bool = False,
    healpix_depths: int | Sequence[int] = [18],
    healpix_column_name: str = "healpix_{depth}",
    ra_column: str = "ra",
    dec_column: str = "dec",
    ra_column_unit: Literal["degrees", "hours", "radians"] = "degrees",
    dec_column_unit: Literal["degrees", "radians"] = "degrees",
    add_row_index_column: bool = False,
    row_index_column_name: str = "id",
    row_index_start: int = 1,
    apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None = None,
) -> int:
    """Converts a single CSV file to Parquet format using the provided schema."""

    csv_file = pathlib.Path(csv_file)

    scan_csv_kwargs = scan_csv_kwargs or {}
    sink_parquet_kwargs = sink_parquet_kwargs or {}

    lf = polars.scan_csv(csv_file, **scan_csv_kwargs)
    lf = _process_lazy_frame(
        lf,
        schema,
        ra_column=ra_column,
        dec_column=dec_column,
        ra_column_unit=ra_column_unit,
        dec_column_unit=dec_column_unit,
        add_healpix_columns=add_healpix_columns,
        healpix_depths=healpix_depths,
        healpix_column_name=healpix_column_name,
        add_row_index_column=add_row_index_column,
        row_index_column_name=row_index_column_name,
        row_index_start=row_index_start,
        apply_callback=apply_callback,
    )

    if ".csv.gz" in csv_file.name:
        stem = csv_file.name[:-7]
    else:
        stem = csv_file.stem

    output_file = output_dir / (stem + ".parquet")
    lf.sink_parquet(output_file, **sink_parquet_kwargs)

    return lazy_frame_row_count(lf)


def _process_lazy_frame(
    lf: polars.LazyFrame,
    schema: Schema,
    ra_column: str = "ra",
    dec_column: str = "dec",
    ra_column_unit: Literal["degrees", "hours", "radians"] = "degrees",
    dec_column_unit: Literal["degrees", "radians"] = "degrees",
    add_healpix_columns: bool = False,
    healpix_depths: int | Sequence[int] = [18],
    healpix_column_name: str = "healpix_{depth}",
    add_row_index_column: bool = False,
    row_index_column_name: str = "id",
    row_index_start: int = 1,
    apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None = None,
) -> polars.LazyFrame:
    """Processes a lazy frame, including calculating the HEALPix column."""

    if ra_column_unit != "degrees":
        ra_column_deg = _generate_column_name(
            f"{ra_column}_deg",
            lf.columns,
        )

        if ra_column_unit == "hours":
            lf = lf.with_columns((polars.col(ra_column) * 15).alias(ra_column_deg))
        elif ra_column_unit == "radians":
            lf = lf.with_columns(
                (polars.col(ra_column) * 180 / numpy.pi).alias(ra_column_deg)
            )

    if dec_column_unit != "degrees":
        dec_column_deg = _generate_column_name(
            f"{dec_column}_deg",
            lf.columns,
        )

        if dec_column_unit == "radians":
            lf = lf.with_columns(
                (polars.col(dec_column) * 180 / numpy.pi).alias(dec_column_deg)
            )

    if add_healpix_columns:
        if isinstance(healpix_depths, int):
            healpix_depths = [healpix_depths]

        for healpix_depth in healpix_depths:
            lf = lf.with_columns(
                get_healpix(
                    polars.col(ra_column),
                    polars.col(dec_column),
                    depth=healpix_depth,
                ).alias(healpix_column_name.format(depth=healpix_depth))
            )

    if add_row_index_column:
        lf = lf.with_row_index(
            name=row_index_column_name,
            offset=row_index_start,
        )

    # Apply any casting to the source columns as defined in the schema.
    schema_columns = schema.schema.columns
    lf_columns = lf.collect_schema().names()
    for lf_col in lf_columns:
        for sch_col, sch_col_data in schema_columns.items():
            source_cat = sch_col_data.source_catalogue_params
            source_col = (source_cat.column_name if source_cat else None) or sch_col
            if lf_col == source_col and source_cat:
                cast_to = source_cat.cast_to
                if cast_to and source_cat.dtype:
                    cast_to_dtype = get_polars_dtype_from_string(cast_to)
                    lf = lf.with_columns(polars.col(lf_col).cast(cast_to_dtype))

    if apply_callback is not None:
        if not isinstance(apply_callback, (list, tuple)):
            apply_callback = [apply_callback]

        for callback_func in apply_callback:
            if isinstance(callback_func, str):
                if callback_func not in callbacks:
                    raise ValueError(
                        f"Callback function '{callback_func}' is not defined. "
                        f"Available callbacks: {list(callbacks.keys())}"
                    )

                lf = callbacks[callback_func](lf)
            else:
                lf = callback_func(lf)  # ty:ignore[call-non-callable]

    return lf


def _generate_column_name(base_name: str, columns: Sequence[str]) -> str:
    """Generates a column name, checking againt a list of existing columns."""

    if base_name not in columns:
        return base_name

    for ii in range(1, 1000):
        new_name = f"{base_name}_{ii}"
        if new_name not in columns:
            return new_name

    raise ValueError(f"Could not generate a unique column name based on {base_name!r}.")
