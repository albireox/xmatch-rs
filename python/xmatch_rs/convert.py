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

from xmatch_rs import console
from xmatch_rs.expressions import get_healpix
from xmatch_rs.schema import Schema
from xmatch_rs.tools import lazy_frame_row_count


__all__: list[str] = ["convert_csv_to_parquet"]


def convert_csv_to_parquet(
    csv_files: Sequence[str | pathlib.Path] | str | pathlib.Path,
    output_dir: str | pathlib.Path,
    schema: Schema | str | pathlib.Path | dict,
    allow_extra_columns: bool = False,
    mode: Literal["direct", "partition"] = "direct",
    partition_key: str | Sequence[str] | None = None,
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
    apply_callback: Callable[[polars.LazyFrame], polars.LazyFrame] | None = None,
    parallelise: bool = True,
    n_workers: int | None = None,
    verbose: bool = False,
    scan_csv_kwargs: dict | None = None,
    sink_parquet_kwargs: dict | None = None,
    show_progress: bool = True,
) -> None:
    """Converts CSV files to Parquet format using the provided schema.

    Parameters
    ----------
    csv_files
        A sequence of paths to the CSV files to convert to Parquet. Can also be
        a single path to a CSV file or a directory containing CSV files, in which
        scase all CSV files in the directory will be converted.
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
        in ``"partition"`` mode.
    parallelise
        Whether to parallelise the conversion. Only applicable if ``mode`` is
        ``"direct"``.
    n_workers
        The number of workers to use for parallelisation.
    verbose
        Whether to print verbose output.
    scan_csv_kwargs
        Additional keyword arguments to pass to `polars.scan_csv`.
    sink_parquet_kwargs
        Additional keyword arguments to pass to `polars.LazyFrame.sink_parquet`.
    show_progress
        Whether to show a progress bar during the conversion. Only applicable
        if ``mode`` is ``"direct"``.

    """

    # Retrieve the schema if a path or name is provided.
    if not isinstance(schema, Schema):
        schema = Schema(schema)

    schema_dtypes = schema.get_dtypes(exclude_xmatch_columns=True)

    # Prepare the list of CSV files to convert.
    if isinstance(csv_files, (str, pathlib.Path)):
        csv_path = pathlib.Path(csv_files)
        if csv_path.is_dir():
            csv_files = list(csv_path.glob("*.csv"))
        else:
            csv_files = [csv_path]
    else:
        csv_files = [pathlib.Path(file_) for file_ in csv_files]

    # Prepare the scan and sink kwargs with some defaults.
    scan_csv_kwargs = scan_csv_kwargs or {}
    sink_parquet_kwargs = sink_parquet_kwargs or {}

    if allow_extra_columns:
        scan_csv_kwargs["schema_overrides"] = schema_dtypes
    else:
        scan_csv_kwargs["schema"] = schema_dtypes

    source_params = schema.schema.source_params
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
        console.print(
            "[yellow]No source_params found in schema. Using default "
            "parameters for reading CSV files.[/]"
        )

    # Set values for scan_csv_kwargs.
    if "comment_prefix" not in scan_csv_kwargs:
        scan_csv_kwargs["comment_prefix"] = "#"
    if "null_values" not in scan_csv_kwargs:
        scan_csv_kwargs["null_values"] = ["null", "", "NULL"]

    console.print(f"[blue]Converting {len(csv_files)} CSV files to Parquet ... [/]")

    if verbose:
        if schema.schema_file:
            console.print(f"[bright_black]Using schema file: {schema.schema_file!s}[/]")

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "direct":
        proc = multiprocessing.get_context("spawn")

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
        pbar_task = pbar.add_task("Converting CSV to Parquet ...", total=len(csv_files))

        if show_progress:
            pbar.start()

        if parallelise:
            if add_row_index_column:
                raise ValueError(
                    "Cannot add a row index column when parallelising "
                    "the conversion in direct mode."
                )

            _imap_partial = partial(
                _convert_csv_to_parquet_direct,
                output_dir=output_dir,
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
                apply_callback=apply_callback,
            )

            with proc.Pool(n_workers if parallelise else 1) as pool:
                for _ in pool.imap_unordered(_imap_partial, csv_files):
                    pbar.advance(pbar_task)

        else:
            row_index_start = 1
            for csv_file in sorted(csv_files):
                n_rows = _convert_csv_to_parquet_direct(
                    csv_file=csv_file,
                    output_dir=output_dir,
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
                    row_index_start=row_index_start,
                    apply_callback=apply_callback,
                )
                row_index_start += n_rows
                pbar.advance(pbar_task)

        pbar.stop()

    elif mode == "partition":
        lf = polars.scan_csv(csv_files, **scan_csv_kwargs)

        lf = _process_lazy_frame(
            lf,
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
            max_rows_per_file=partition_max_rows_per_file,
            approximate_bytes_per_file=partition_approximate_bytes_per_file,
        )

        lf.sink_parquet(partition_by, **sink_parquet_kwargs)

    console.print(f"[green]Conversion complete. Parquet files saved to {output_dir!s}.")


def _convert_csv_to_parquet_direct(
    csv_file: pathlib.Path,
    output_dir: pathlib.Path,
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
    apply_callback: Callable[[polars.LazyFrame], polars.LazyFrame] | None = None,
) -> int:
    """Converts a single CSV file to Parquet format using the provided schema."""

    scan_csv_kwargs = scan_csv_kwargs or {}
    sink_parquet_kwargs = sink_parquet_kwargs or {}

    lf = polars.scan_csv(csv_file, **scan_csv_kwargs)
    lf = _process_lazy_frame(
        lf,
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

    output_file = output_dir / (csv_file.stem + ".parquet")
    lf.sink_parquet(output_file, **sink_parquet_kwargs)

    return lazy_frame_row_count(lf)


def _process_lazy_frame(
    lf: polars.LazyFrame,
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
    apply_callback: Callable[[polars.LazyFrame], polars.LazyFrame] | None = None,
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

    if apply_callback:
        lf = apply_callback(lf)

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
