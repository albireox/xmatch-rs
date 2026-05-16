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
from functools import partial

from typing import TYPE_CHECKING, Iterable, Literal, Sequence

import numpy
import polars
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from xmatch_rs import console
from xmatch_rs.convert.callbacks import callbacks
from xmatch_rs.expressions import get_healpix
from xmatch_rs.schema import Schema
from xmatch_rs.tools import (
    get_polars_dtype_from_string,
    get_xmatch_rs_logger,
    lazy_frame_row_count,
    parse_path,
)


if TYPE_CHECKING:
    from sdsstools.logger import SDSSLogger

    from xmatch_rs.types import APPLY_CALLBACK_T, DATA_PATH_T, FILE_PATH_PROVIDER_T

__all__ = ["convert_csv_to_parquet", "convert_to_parquet"]


def convert_csv_to_parquet(
    csv_files: DATA_PATH_T,
    output_path: str | pathlib.Path,
    schema: Schema | str | pathlib.Path | dict,
    allow_extra_columns: bool = False,
    mode: Literal["direct", "partition"] = "direct",
    partition_process_mode: Literal["single", "combined"] = "combined",
    partition_key: str | list[str] | None = None,
    file_path_provider: FILE_PATH_PROVIDER_T | None = None,
    partition_max_rows_per_file: int | None = None,
    partition_approximate_bytes_per_file: int | Literal["auto"] | None = "auto",
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
    partition_process_mode
        The mode to use to process the data frame for partitioning. If ``"single"`,
        each data frame is processed independently and appended to the partition
        output. This requires providing a ``file_path_provider`` to prevent overwriting
        the output files. If ``"combined"``, all data frames are combined into a single
        data frame before processing and partitioning. This is the default but may
        require more memory.
    partition_key
        The key(s) to use for partitioning.
    partition_max_rows_per_file
        The maximum number of rows per partitioned file.
    partition_approximate_bytes_per_file
        The approximate number of bytes per partitioned file.
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

    output_path = pathlib.Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        if not output_path.is_dir():
            raise ValueError(f"Output path {output_path!s} is not a directory.")

    log = log or get_xmatch_rs_logger("xmatch_rs.convert")

    # Retrieve the schema if a path or name is provided.
    if not isinstance(schema, Schema):
        schema = Schema(schema)

    polars_schema = schema.get_polars_schema(source_columns=True)

    # Prepare the list of CSV files to convert.
    if isinstance(csv_files, (str, pathlib.Path)):
        csv_files = [csv_files]

    csv_paths = parse_path(
        csv_files,
        extension=r"\.csv.*",
        recursive=False,
    )

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

    lfs: list[polars.LazyFrame] = []
    output_paths: list[pathlib.Path] = []
    for csv_path in csv_paths:
        lf = polars.scan_csv(csv_path, **scan_csv_kwargs)

        if ".csv.gz" in csv_path.name:
            stem = csv_path.name[:-7]
        else:
            stem = csv_path.stem

        output_file = stem + ".parquet"

        lfs.append(lf)
        output_paths.append(output_path / output_file)

    convert_to_parquet(
        lfs,
        output_paths=output_paths if mode == "direct" else output_path,
        schema=schema,
        mode=mode,
        partition_process_mode=partition_process_mode,
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
    | Iterable[polars.DataFrame | polars.LazyFrame],
    output_paths: str | pathlib.Path | Iterable[str | pathlib.Path],
    schema: Schema | str | pathlib.Path | dict,
    mode: Literal["direct", "partition"] = "direct",
    partition_process_mode: Literal["single", "combined"] = "combined",
    partition_key: str | list[str] | None = None,
    file_path_provider: FILE_PATH_PROVIDER_T | None = None,
    partition_max_rows_per_file: int | None = None,
    partition_approximate_bytes_per_file: int | Literal["auto"] | None = "auto",
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
    output_paths
        The directory where the converted Parquet files will be saved. Alternatively,
        a list of paths to the output Parquet files, in which case each data frame
        will be saved to the corresponding path.
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
    partition_process_mode
        The mode to use to process the data frame for partitioning. If ``"single"``,
        each data frame is processed independently and appended to the partition
        output. This requires providing a ``file_path_provider`` to prevent overwriting
        the output files. If ``"combined"``, all data frames are combined into a single
        data frame before processing and partitioning. This is the default but may
        require more memory.
    partition_key
        The key(s) to use for partitioning.
    file_path_provider
        An optional function that takes the partition key value(s) and returns a
        file path to write the partition to.
    partition_max_rows_per_file
        The maximum number of rows per partitioned file.
    partition_approximate_bytes_per_file
        The approximate number of bytes per partitioned file.
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
        Whether to parallelise the conversion.
    n_workers
        The number of workers to use for parallelisation.
    scan_csv_kwargs
        Additional keyword arguments to pass to `polars.scan_csv`.
    sink_parquet_kwargs
        Additional keyword arguments to pass to `polars.LazyFrame.sink_parquet`.
    show_progress
        Whether to show a progress bar during the conversion. Only applicable
        if ``mode`` is ``"direct"`` or ``"partition"`` with
        ``partition_process_mode="single"``.
    log
        An optional logger to use for logging messages during the conversion.

    """

    # Create logger or use the one provided.
    log = log or get_xmatch_rs_logger("xmatch_rs.convert")

    # Retrieve the schema if a path or name is provided.
    if not isinstance(schema, Schema):
        schema = Schema(schema)

    if isinstance(dfs, (polars.DataFrame, polars.LazyFrame)):
        lfs = [dfs.lazy()]
    else:
        lfs = [df.lazy() for df in dfs]

    # Create a progress bar.
    pbar = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )
    pbar_task: int | None = None

    # Handle parallelisation options.
    if n_workers == 1:
        parallelise = False
    elif parallelise and n_workers is None:
        n_workers = multiprocessing.cpu_count()

    log.debug(f"Converting files using mode {mode!r}")

    if mode == "direct":
        if isinstance(output_paths, (str, pathlib.Path)):
            # If the output paths are not defined, use a template path.
            base_path = pathlib.Path(output_paths)
            n_zeros = len(str(len(lfs)))
            output_paths = [
                base_path / f"part_{ii:0{n_zeros}d}.parquet" for ii in range(len(lfs))
            ]
        else:
            output_paths = [pathlib.Path(p) for p in output_paths]
            if len(output_paths) != len(lfs):
                raise ValueError(
                    "If output_paths is a list of paths, it must have the same "
                    "length as the number of data frames."
                )

        # Add a task if show_progress=True
        if show_progress:
            pbar_task = pbar.add_task(
                "Processing ...",
                total=len(lfs),
            )
            pbar.start()

        # Create a partial function for the conversion to Parquet.
        _imap_partial = partial(
            _convert_to_parquet_direct,
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

        if parallelise:
            if add_row_index_column:
                raise ValueError(
                    "Cannot add a row index column when parallelising "
                    "the conversion in direct mode."
                )

            proc = multiprocessing.get_context("spawn")

            log.debug(f"Parallelising conversion with {n_workers} workers ...")

            with proc.Pool(n_workers) as pool:
                for _ in pool.imap_unordered(
                    lambda args: _imap_partial(*args),
                    zip(lfs, output_paths),
                ):
                    if pbar_task is not None:
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
                if pbar_task is not None:
                    pbar.advance(pbar_task)

        pbar.stop()

    elif mode == "partition":
        if not isinstance(output_paths, (str, pathlib.Path)):
            raise ValueError("When mode='partition', output_paths must be a directory.")

        base_path = pathlib.Path(output_paths)

        log.debug(
            f"Partitioning by {partition_key!r} with mode={partition_process_mode!r}, "
            f"max_rows_per_file={partition_max_rows_per_file} and "
            f"approximate_bytes_per_file={partition_approximate_bytes_per_file}"
        )

        if partition_process_mode == "single":
            if show_progress:
                pbar_task = pbar.add_task(
                    "Processing ...",
                    total=len(lfs),
                )
                pbar.start()
        else:
            lfs = [polars.concat(lfs)]

        sink_parquet_kwargs = sink_parquet_kwargs or {}
        partition_by_kwargs = {
            "base_path": base_path,
            "key": partition_key,
            "file_path_provider": file_path_provider,
            "include_key": partition_include_key,
            "max_rows_per_file": partition_max_rows_per_file,
            "approximate_bytes_per_file": partition_approximate_bytes_per_file,
        }

        _process_lazy_frame_partial = partial(
            _process_lazy_frame,
            schema=schema,
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
            sink_parquet=True,
            partition_by_kwargs=partition_by_kwargs,
            sink_parquet_kwargs=sink_parquet_kwargs,
        )

        if parallelise:
            proc = multiprocessing.get_context("spawn")

            log.debug(f"Parallelising conversion with {n_workers} workers ...")

            with proc.Pool(n_workers) as pool:
                for _ in pool.imap_unordered(_process_lazy_frame_partial, lfs):
                    if pbar_task is not None:
                        pbar.advance(pbar_task)

        else:
            for lf in lfs:
                _process_lazy_frame_partial(lf)
                if pbar_task is not None:
                    pbar.advance(pbar_task)

    else:
        raise ValueError("Invalid mode. Must be 'direct' or 'partition'.")

    if pbar_task is not None:
        pbar.stop()


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
    sink_parquet: bool = False,
    output_path: pathlib.Path | None = None,
    partition_by_kwargs: dict | None = None,
    sink_parquet_kwargs: dict | None = None,
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

    if sink_parquet:
        if output_path:
            lf = lf.sink_parquet(output_path, **(sink_parquet_kwargs or {}))
        elif partition_by_kwargs:
            partition_by = polars.PartitionBy(**partition_by_kwargs)
            lf = lf.sink_parquet(partition_by, **(sink_parquet_kwargs or {}))
        else:
            raise ValueError(
                "Must provide either output_path or partition_by_kwargs "
                "to sink to Parquet."
            )

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
