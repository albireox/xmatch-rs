#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-07-03
# @Filename: astropy.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


from __future__ import annotations

import functools
import multiprocessing
import pathlib

from typing import TYPE_CHECKING, Literal, Sequence

import polars
from astropy.table import Table
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from xmatch_rs import console
from xmatch_rs.schema.schema import ASTROPY_FORMATS, Schema
from xmatch_rs.tools import get_xmatch_rs_logger, list_files


if TYPE_CHECKING:
    from sdsstools.logger import SDSSLogger

    from xmatch_rs.types import APPLY_CALLBACK_T, DATA_PATH_T, FILE_PATH_PROVIDER_T


__all__ = ["read_astropy_to_dataframe", "convert_astropy_to_parquet"]


def read_astropy_to_dataframe(
    path: str | pathlib.Path,
    format: str,
) -> polars.DataFrame:
    """Reads a file using ``astropy.table.Table`` and returns a Polars DataFrame."""

    valid_formats = [ff.name.lower() for ff in ASTROPY_FORMATS]
    if format.lower() not in valid_formats:
        raise ValueError(f"Invalid format {format!r}.")

    table = Table.read(path, format=format)

    return table.to_df("polars")


def convert_astropy_to_parquet(
    files: DATA_PATH_T,
    format: str,
    output_path: str | pathlib.Path,
    schema: Schema | str | pathlib.Path | dict,
    mode: Literal["direct", "partition"] = "direct",
    partition_key: str | list[str] | None = None,
    file_path_provider: FILE_PATH_PROVIDER_T | None = None,
    partition_max_rows_per_file: int | None = None,
    partition_approximate_bytes_per_file: int | Literal["auto"] | None = "auto",
    partition_include_key: bool = False,
    add_healpix_columns: bool = True,
    healpix_orders: Sequence[int] = [29],
    healpix_column_name: str = "_healpix_{order}",
    ra_column: str = "ra",
    dec_column: str = "dec",
    ra_column_unit: Literal["degrees", "hours", "radians"] = "degrees",
    dec_column_unit: Literal["degrees", "radians"] = "degrees",
    apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None = None,
    parallelise: bool = False,
    n_workers: int | None = None,
    show_progress: bool = False,
    log: SDSSLogger | None = None,
) -> None:
    """Converts files to Parquet by first reading them with ``astropy.table.Table``.

    Parameters
    ----------
    files
        A sequence of paths to the files to convert to Parquet. Can also be
        a single path to a file or a directory containing files, in which
        case all files in the directory will be converted. Glob patters are
        supported.
    format
        The format to use for reading the files with ``astropy.table.Table``.
        Must be one of the `supported text formats <https://docs.astropy.org/en/latest/io/unified_table_text.html>`_.
    output_path
        The directory where the converted Parquet files will be saved. The file
        names will be the same as the input files, but with a ``.parquet``
        extension. In parition mode, the files will be saved according to the
        file path provider.
    schema
        The schema to use for the conversion, or a path to the YAML file
        containing the schema.
    mode
        The conversion mode. If ``"direct``, each files will be converted
        to a Parquet files. If ``"partition"``, the files will be partitioned
        according to the provided parameters. The latter requires passing a
        ``file_path_provider`` to prevent overwriting the output files.
    file_path_provider
        An optional function that takes the partition key value(s) and returns a
        file path to write the partition to.
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
    show_progress
        Whether to show a progress bar during the conversion.
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

    # Prepare the list of CSV files to convert.
    if isinstance(files, (str, pathlib.Path)):
        files = [files]

    paths = list_files(files)

    if len(paths) == 0:
        raise ValueError("No files found to convert.")

    log.info(f"Converting {len(paths)} files to Parquet ...")

    if schema.schema_file:
        log.debug(f"Using schema file: {schema.schema_file!s}")

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

    if show_progress:
        pbar_task = pbar.add_task("Converting files ...", total=len(paths))
        pbar.start()

    if mode == "partition" and file_path_provider is None:
        raise ValueError("A file path provider must be provided in partition mode.")

    # Handle parallelisation options.
    if parallelise and n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if n_workers is None:
        n_workers = 1

    _process_file_partial = functools.partial(
        _process_file,
        format=format,
        schema=schema,
        output_path=output_path,
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
        apply_callback=apply_callback,
        log=log,
    )

    with multiprocessing.Pool(n_workers) as pool:
        for _ in pool.imap(_process_file_partial, paths):
            if show_progress and pbar_task is not None:
                pbar.update(pbar_task, advance=1)

    if show_progress and pbar_task is not None:
        pbar.stop()

    log.info(f"Conversion complete. Parquet files saved to {output_path!s}.")


def _process_file(
    path: pathlib.Path,
    format: str,
    schema: Schema,
    output_path: pathlib.Path,
    mode: Literal["direct", "partition"],
    partition_key: str | list[str] | None,
    file_path_provider: FILE_PATH_PROVIDER_T | None,
    partition_max_rows_per_file: int | None,
    partition_approximate_bytes_per_file: int | Literal["auto"] | None,
    partition_include_key: bool,
    add_healpix_columns: bool,
    healpix_orders: Sequence[int],
    healpix_column_name: str,
    ra_column: str,
    dec_column: str,
    ra_column_unit: Literal["degrees", "hours", "radians"],
    dec_column_unit: Literal["degrees", "radians"],
    apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None,
    log: SDSSLogger,
) -> None:
    """Processes a single file and converts it to Parquet format."""

    from xmatch_rs.convert.convert import _process_lazy_frame

    try:
        df = read_astropy_to_dataframe(path, format=format)
    except Exception as err:
        log.error(f"Error reading file {path!s}: {err!s}")
        return

    lf = _process_lazy_frame(
        df.lazy(),
        schema=schema,
        ra_column=ra_column,
        dec_column=dec_column,
        ra_column_unit=ra_column_unit,
        dec_column_unit=dec_column_unit,
        add_healpix_columns=add_healpix_columns,
        healpix_orders=healpix_orders,
        healpix_column_name=healpix_column_name,
        apply_callback=apply_callback,
        sink_parquet=False,
    )

    if mode == "direct":
        all_suffixes = "".join(path.suffixes)
        true_stem = path.name.removesuffix(all_suffixes)

        output_file = output_path / (true_stem + ".parquet")
        lf.sink_parquet(output_file, engine="streaming")

        return

    lf.sink_parquet(
        polars.PartitionBy(
            base_path=output_path,
            key=partition_key,
            file_path_provider=file_path_provider,
            include_key=partition_include_key,
            max_rows_per_file=partition_max_rows_per_file,
            approximate_bytes_per_file=partition_approximate_bytes_per_file,
        ),
        engine="streaming",
    )
