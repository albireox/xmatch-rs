#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-06-06
# @Filename: csv.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import TYPE_CHECKING, Literal, Sequence

import polars

from xmatch_rs.schema import Schema
from xmatch_rs.tools import get_xmatch_rs_logger, list_files


if TYPE_CHECKING:
    from sdsstools.logger import SDSSLogger

    from xmatch_rs.types import APPLY_CALLBACK_T, DATA_PATH_T, FILE_PATH_PROVIDER_T


__all__ = ["convert_csv_to_parquet"]


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
        The schema to use for the conversion, or a path to the YAML file
        containing the schema.
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

    from xmatch_rs.convert import convert_to_parquet

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

    polars_schema = schema.get_polars_schema(source_catalogue=True)

    # Prepare the list of CSV files to convert.
    if isinstance(csv_files, (str, pathlib.Path)):
        csv_files = [csv_files]

    csv_paths = list_files(
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

    source_format = schema.model.source_catalogue.format
    if source_format != "csv":
        raise ValueError("Schema for catalogue has source_catalogue.format != 'csv'.")

    source_params = schema.model.source_catalogue.read_params
    if source_params is not None:
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
