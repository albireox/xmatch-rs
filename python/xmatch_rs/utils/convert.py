#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-03-17
# @Filename: convert.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import json
import pathlib

from typing import Literal, Sequence

from xmatch_rs import console

from .schema import get_schema


__all__: list[str] = ["convert_csv_to_parquet"]


def convert_csv_to_parquet(
    csv_files: Sequence[str | pathlib.Path] | str | pathlib.Path,
    output_dir: str | pathlib.Path,
    schema_file: str | pathlib.Path,
    mode: Literal["direct", "partition"] = "direct",
    partition_key: str | Sequence[str] | None = None,
    partition_max_rows_per_file: int | None = None,
    partition_approximate_bytes_per_file: int | Literal["auto"] | None = "auto",
    parallelise: bool = False,
    n_workers: int | None = None,
    verbose: bool = False,
    scan_csv_kwargs: dict | None = None,
    sink_csv_kwargs: dict | None = None,
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
    schema_file
        The path to the schema file. It can also be the name of a schema in the
        ``schemas/`` directory.
    mode
        The conversion mode. If ``"direct``, each CSV files will be converted
        to a Parquet files. If ``"partition"``, the CSV files will be partitioned
        according to the provided parameters.
    partition_key
        The key(s) to use for partitioning.
    partition_max_rows_per_file
        The maximum number of rows per partitioned file.
    partition_approximate_bytes_per_file
        The approximate number of bytes per partitioned file.
    parallelise
        Whether to parallelise the conversion. Only applicable if ``mode`` is
        ``"direct"``.
    n_workers
        The number of workers to use for parallelisation.
    verbose
        Whether to print verbose output.
    scan_csv_kwargs
        Additional keyword arguments to pass to `polars.scan_csv`.
    sink_csv_kwargs
        Additional keyword arguments to pass to `polars.LazyFrame.sink_parquet`.

    """

    schema = get_schema(schema_file)

    if isinstance(csv_files, (str, pathlib.Path)):
        csv_path = pathlib.Path(csv_files)
        if csv_path.is_dir():
            csv_files = list(csv_path.glob("*.csv"))
        else:
            csv_files = [csv_path]
    else:
        csv_files = [pathlib.Path(file_) for file_ in csv_files]

    console.print(f"[blue]Converting {len(csv_files)} CSV files to Parquet ... [/]")

    if verbose:
        console.print("[bright_black]Using schema:[/]")
        console.print("[bright_black]" + json.dumps(schema, indent=2) + "[/]")
