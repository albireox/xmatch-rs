#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-03-17
# @Filename: convert.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import pathlib
import re
import time
from functools import partial

from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy
import polars
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from sdsstools.logger import SDSSLogger

from xmatch_rs import console
from xmatch_rs.convert import get_hats_file_provider
from xmatch_rs.convert.callbacks import callbacks
from xmatch_rs.expressions import get_healpix
from xmatch_rs.schema import Schema
from xmatch_rs.tools import (
    get_polars_dtype_from_string,
    get_xmatch_rs_logger,
    lazy_frame_row_count,
    parse_path,
)


__all__: list[str] = ["convert_csv_to_parquet", "convert_to_parquet"]


CALLBACK_FUNC_T = Callable[[polars.LazyFrame], polars.LazyFrame]
APPLY_CALLBACK_T = CALLBACK_FUNC_T | str

FILE_PATH_PROVIDER_T = Callable[[polars.FileProviderArgs], pathlib.Path]


def convert_to_hats(
    path: list[str | pathlib.Path] | str | pathlib.Path,
    schema: Schema | str | pathlib.Path | dict,
    output_path: str | pathlib.Path,
    format: Literal["csv", "parquet"] | None = None,
    hats_initial_order: int = 6,
    hats_max_order: int | None = None,
    hats_max_rows_per_pixel: int | None = None,
    partition_max_rows_per_file: int | None = None,
    partition_approximate_bytes_per_file: int | Literal["auto"] | None = None,
    apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None = None,
    convert_kwargs: dict | None = None,
    log: SDSSLogger | None = None,
    quiet: bool = False,
) -> None:
    """Converts a dataset to the HATS format.

    Parameters
    ----------
    path
        The path(s) to the input dataset. Can be a single path or a list of paths.
    schema
        The schema of the input dataset. Can be a Schema object, a path to a schema
        file, or a dictionary representing the schema.
    output_path
        The path where the output HATS dataset will be saved.
    format
        The format of the output files. If not provided, it will be inferred from
        the files in ``path``.
    hats_initial_order
        The initial HEALPix order to use for partitioning the data. Defaults to 6.
    hats_max_order
        The maximum HEALPix order to use for partitioning the data. If not
        provided, the tiling process will continue until all pixels contain
        fewer than ``hats_max_rows_per_file`` rows.
    hats_max_rows_per_pixel
        The maximum number of rows per pixel. If set to ``None``, no adaptive
        tiling will be performed and the data will be partitioned using the
        ``hats_initial_order``.
    partition_max_rows_per_file
        The maximum number of rows per partitioned file.
    partition_approximate_bytes_per_file
        The approximate number of bytes per partitioned file. Defaults to 1 GB.
    apply_callback
        A callback function or list of callback functions to apply to the data
        before saving. Each callback should take a Polars LazyFrame as input
        and return a modified LazyFrame.
    convert_kwargs
        Additional keyword arguments to pass to the conversion function.
    log
        An optional logger to use for logging the conversion process.
    quiet
        If ``True``, suppresses logging output. Defaults to ``False``.

    """

    # Get logger
    log = log or get_xmatch_rs_logger(log_level=40 if quiet else 20)

    # Get the file provider for HATS files.
    hats_file_provider = get_hats_file_provider(
        hats_initial_order,
        column_name=f"_healpix_{hats_initial_order}",
    )

    # Determine the file format if not provided.
    if format is None:
        files = parse_path(path)
        sample_file = files[0]
        if re.match(r".*\.csv\.?[gz|bz2]?$", sample_file.name):
            log.info("Inferred file format as CSV based on file extension.")
            format = "csv"
        elif re.match(r".*\.parquet$", sample_file.name):
            log.info("Inferred file format as Parquet based on file extension.")
            format = "parquet"
        else:
            raise ValueError(
                "Could not infer file format from file extension. "
                "Please specify the format explicitly.",
            )

    # Perform initial conversion to HATS format using the initial HEALPix order.
    t0 = time.time()

    if format == "csv":
        log.info("Converting CSV files to HATS.")
        log.info(f"Performing initial convertion to order {hats_initial_order}.")

        convert_csv_to_parquet(
            csv_files=path,
            schema=schema,
            output_path=output_path,
            mode="partition",
            partition_key=f"_healpix_{hats_initial_order}",
            file_path_provider=hats_file_provider,
            partition_max_rows_per_file=partition_max_rows_per_file,
            partition_approximate_bytes_per_file=partition_approximate_bytes_per_file,
            partition_include_key=False,
            healpix_orders=[29, hats_initial_order],
            healpix_column_name="_healpix_{order}",
            apply_callback=apply_callback,
            log=log,
            **(convert_kwargs or {}),
        )

    elif format == "parquet":
        log.info("Converting Parquet files to HATS.")
        log.info(f"Performing initial convertion to order {hats_initial_order}.")

        if isinstance(path, (str, pathlib.Path)):
            path = [path]

        lf = polars.scan_parquet([pathlib.Path(p) for p in path])

        convert_to_parquet(
            lf,
            schema=schema,
            output_path=output_path,
            mode="partition",
            partition_key=f"_healpix_{hats_initial_order}",
            file_path_provider=hats_file_provider,
            partition_max_rows_per_file=partition_max_rows_per_file,
            partition_approximate_bytes_per_file=partition_approximate_bytes_per_file,
            partition_include_key=False,
            healpix_orders=[29, hats_initial_order],
            healpix_column_name="_healpix_{order}",
            apply_callback=apply_callback,
            log=log,
            **(convert_kwargs or {}),
        )

    else:
        raise ValueError(
            f"Invalid format {format!r}. Currently supported formats "
            "are 'csv' and 'parquet'.",
        )

    t1 = time.time()
    log.info(f"Initial conversion completed in {t1 - t0:.1f} seconds.")


def convert_csv_to_parquet(
    csv_files: list[str | pathlib.Path] | str | pathlib.Path,
    output_path: str | pathlib.Path,
    schema: Schema | str | pathlib.Path | dict,
    allow_extra_columns: bool = False,
    mode: Literal["direct", "partition"] = "direct",
    partition_key: str | list[str] | None = None,
    file_path_provider: FILE_PATH_PROVIDER_T | None = None,
    partition_max_rows_per_file: int | None = None,
    partition_approximate_bytes_per_file: int | Literal["auto"] | None = None,
    partition_include_key: bool = False,
    add_healpix_columns: bool = True,
    healpix_orders: int | Sequence[int] = [29],
    healpix_column_name: str = "_healpix_{order}",
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
    output_path
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
    partition_include_key
        Whether to include the partition key in the output Parquet files.
    add_healpix_columns
        Whether to add a HEALPix column to the output Parquet files. If ``True``,
        the HEALPix column will be added using the provided RA and Dec columns and
        the specified order.
    healpix_orders
        A sequence of orders to use for the HEALPix column(s).
    healpix_column_name
        The name to use for the HEALPix column. Can include ``{order}``, which
        will be replaced by the actual order used (this is required if multiple
        orders are provided).
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

    log = log or get_xmatch_rs_logger("xmatch_rs.convert")

    # Retrieve the schema if a path or name is provided.
    if not isinstance(schema, Schema):
        schema = Schema(schema)

    polars_schema = schema.get_polars_schema(source_columns=True)

    # Prepare the list of CSV files to convert.
    if isinstance(csv_files, (str, pathlib.Path)):
        csv_files = [csv_files]

    csv_paths = parse_path(csv_files, extension=r"\.csv.*")

    if len(csv_paths) == 0:
        raise ValueError("No CSV files found to convert.")

    # Prepare the scan kwargs with some defaults.
    scan_csv_kwargs = scan_csv_kwargs or {}

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

    lfs: dict[str, polars.LazyFrame] = {}
    for csv_path in csv_paths:
        lf = polars.scan_csv(csv_path, **scan_csv_kwargs)

        if ".csv.gz" in csv_path.name:
            stem = csv_path.name[:-7]
        else:
            stem = csv_path.stem

        output_file = stem + ".parquet"

        lfs[output_file] = lf

    convert_to_parquet(
        lfs,
        output_path=output_path,
        schema=schema,
        mode=mode,
        partition_key=partition_key,
        file_path_provider=file_path_provider,
        partition_max_rows_per_file=partition_max_rows_per_file,
        partition_approximate_bytes_per_file=partition_approximate_bytes_per_file,
        partition_include_key=partition_include_key,
        add_healpix_columns=add_healpix_columns,
        healpix_orders=healpix_orders,
        healpix_column_name=healpix_column_name,
        ra_column=ra_column,
        dec_column=dec_column,
        ra_column_unit=ra_column_unit,
        dec_column_unit=dec_column_unit,
        add_row_index_column=add_row_index_column,
        row_index_column_name=row_index_column_name,
        apply_callback=apply_callback,
        parallelise=parallelise,
        n_workers=n_workers,
        sink_parquet_kwargs=sink_parquet_kwargs,
        show_progress=show_progress,
        log=log,
    )

    log.info(f"Conversion complete. Parquet files saved to {output_path!s}.")


def convert_to_parquet(
    dfs: polars.DataFrame
    | polars.LazyFrame
    | Iterable[polars.DataFrame | polars.LazyFrame]
    | Mapping[str, polars.DataFrame | polars.LazyFrame],
    output_path: str | pathlib.Path,
    schema: Schema | str | pathlib.Path | dict,
    mode: Literal["direct", "partition"] = "direct",
    partition_key: str | list[str] | None = None,
    file_path_provider: FILE_PATH_PROVIDER_T | None = None,
    partition_max_rows_per_file: int | None = None,
    partition_approximate_bytes_per_file: int | Literal["auto"] | None = 1000_000_000,
    partition_include_key: bool = False,
    add_healpix_columns: bool = True,
    healpix_orders: int | Sequence[int] = [29],
    healpix_column_name: str = "_healpix_{order}",
    ra_column: str = "ra",
    dec_column: str = "dec",
    ra_column_unit: Literal["degrees", "hours", "radians"] = "degrees",
    dec_column_unit: Literal["degrees", "radians"] = "degrees",
    add_row_index_column: bool = False,
    row_index_column_name: str = "id",
    apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None = None,
    parallelise: bool = False,
    n_workers: int | None = None,
    sink_parquet_kwargs: dict | None = None,
    show_progress: bool = False,
    log: SDSSLogger | None = None,
) -> None:
    """Convert a list of DataFrames/LazyFrames to Parquet files.

    Parameters
    ----------
    dfs
        A list of data frames or lazy frames to convert to Parquet.
        It is also possible to provide a dictionary of data frames, in which
        case the keys must be the stem of the output Parquet file to write for
        that data frame (ignored for partition mode).
    output_path
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
    partition_include_key
        Whether to include the partition key in the output Parquet files.
    add_healpix_columns
        Whether to add a HEALPix column to the output Parquet files. If ``True``,
        the HEALPix column will be added using the provided RA and Dec columns and
        the specified order.
    healpix_orders
        A sequence of orders to use for the HEALPix column(s).
    healpix_column_name
        The name to use for the HEALPix column. Can include ``{order}``, which
        will be replaced by the actual order used (this is required if multiple
        orders are provided).
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

    # Create logger or use the one provided.
    log = log or get_xmatch_rs_logger("xmatch_rs.convert")

    # Create the output directory if it doesn't exist.
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Retrieve the schema if a path or name is provided.
    if not isinstance(schema, Schema):
        schema = Schema(schema)

    # Prepare the list of lazy frames and output paths.
    lfs: list[polars.LazyFrame] = []
    output_paths: list[pathlib.Path] = []

    if isinstance(dfs, Mapping):
        lfs = [
            df.lazy()
            for df in dfs.values()
            if isinstance(df, (polars.DataFrame, polars.LazyFrame))
        ]
        output_paths = [(output_path / str(key)) for key in dfs.keys()]

    if isinstance(dfs, (polars.DataFrame, polars.LazyFrame)):
        lfs = [dfs.lazy()]

    # If the output paths are not defined, use a template path.
    lfs = [df.lazy() if isinstance(df, polars.DataFrame) else df for df in lfs]
    n_zeros = len(str(len(lfs)))
    if len(output_paths) == 0:
        output_paths = [
            output_path / f"part_{ii:0{n_zeros}d}.parquet" for ii in range(len(lfs))
        ]

    log.debug(f"Mode: {mode!r}")

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
        pbar_task = pbar.add_task("Converting to Parquet ...", total=len(lfs))

        if show_progress:
            pbar.start()

        _imap_partial = partial(
            _convert_to_parquet_direct,
            output_path=output_path,
            schema=schema,
            sink_parquet_kwargs=sink_parquet_kwargs,
            add_healpix_columns=add_healpix_columns,
            healpix_orders=healpix_orders,
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

            log.debug(f"Parallelising conversion with {n_workers} workers ...")

            with proc.Pool(n_workers) as pool:
                for _ in pool.imap_unordered(_imap_partial, zip(lfs, output_paths)):
                    pbar.advance(pbar_task)

        else:
            row_index_start = 1
            for lf, output_path in zip(lfs, output_paths):
                n_rows = _imap_partial(
                    lf,
                    output_path=output_path,
                    row_index_start=row_index_start,
                )
                row_index_start += n_rows
                pbar.advance(pbar_task)

        pbar.stop()

    elif mode == "partition":
        lf = _process_lazy_frame(
            polars.concat(lfs),
            schema,
            ra_column=ra_column,
            dec_column=dec_column,
            ra_column_unit=ra_column_unit,
            dec_column_unit=dec_column_unit,
            add_healpix_columns=add_healpix_columns,
            healpix_orders=healpix_orders,
            healpix_column_name=healpix_column_name,
            add_row_index_column=add_row_index_column,
            row_index_column_name=row_index_column_name,
            row_index_start=1,
            apply_callback=apply_callback,
        )

        log.debug(
            f"Partitioning by {partition_key!r} with "
            f"max_rows_per_file={partition_max_rows_per_file} and "
            f"approximate_bytes_per_file={partition_approximate_bytes_per_file}"
        )

        partition_by = polars.PartitionBy(
            output_path,
            key=partition_key,
            file_path_provider=file_path_provider,
            include_key=partition_include_key,
            max_rows_per_file=partition_max_rows_per_file,
            approximate_bytes_per_file=partition_approximate_bytes_per_file,
        )

        lf.sink_parquet(partition_by, **(sink_parquet_kwargs or {}))

    else:
        raise ValueError("Invalid mode. Must be 'direct' or 'partition'.")


def _convert_to_parquet_direct(
    lf: polars.LazyFrame,
    output_path: pathlib.Path,
    schema: Schema,
    sink_parquet_kwargs: dict | None = None,
    add_healpix_columns: bool = False,
    healpix_orders: int | Sequence[int] = [18],
    healpix_column_name: str = "_healpix_{order}",
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

    sink_parquet_kwargs = sink_parquet_kwargs or {}

    lf = _process_lazy_frame(
        lf,
        schema,
        ra_column=ra_column,
        dec_column=dec_column,
        ra_column_unit=ra_column_unit,
        dec_column_unit=dec_column_unit,
        add_healpix_columns=add_healpix_columns,
        healpix_orders=healpix_orders,
        healpix_column_name=healpix_column_name,
        add_row_index_column=add_row_index_column,
        row_index_column_name=row_index_column_name,
        row_index_start=row_index_start,
        apply_callback=apply_callback,
    )

    lf.sink_parquet(output_path, **sink_parquet_kwargs)

    return lazy_frame_row_count(lf)


def _process_lazy_frame(
    lf: polars.LazyFrame,
    schema: Schema,
    ra_column: str = "ra",
    dec_column: str = "dec",
    ra_column_unit: Literal["degrees", "hours", "radians"] = "degrees",
    dec_column_unit: Literal["degrees", "radians"] = "degrees",
    add_healpix_columns: bool = False,
    healpix_orders: int | Sequence[int] = [18],
    healpix_column_name: str = "healpix_{order}",
    add_row_index_column: bool = False,
    row_index_column_name: str = "id",
    row_index_start: int = 1,
    apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None = None,
) -> polars.LazyFrame:
    """Processes a lazy frame, including calculating the HEALPix column."""

    columns = lf.collect_schema().names()

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
        if isinstance(healpix_orders, int):
            healpix_orders = [healpix_orders]

        for healpix_order in healpix_orders:
            healpix_column_name_order = healpix_column_name.format(order=healpix_order)
            if healpix_column_name_order not in columns:
                lf = lf.with_columns(
                    get_healpix(
                        polars.col(ra_column),
                        polars.col(dec_column),
                        order=healpix_order,
                    ).alias(healpix_column_name_order)
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
