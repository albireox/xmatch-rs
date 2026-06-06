#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-05-17
# @Filename: index.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import polars

from xmatch_rs.hats.tools import get_dataset_path


__all__ = ["create_index"]


def create_index(
    object_path: pathlib.Path | str,
    index_path: pathlib.Path | str,
    index_column: str,
    extra_columns: list[str] = [],
    check_duplicates: bool = False,
    sort: bool = False,
    max_rows_per_file: int = 10_000_000,
    max_kilobytes_per_file: int = 10 * 1024 * 1024,
) -> None:
    """Creates an index dataset from the object catalogue.

    Parameters
    ----------
    object_path
        Path to the object catalogue.
    index_path
        Path where the index dataset will be created.
    index_column
        The column to index on.
    extra_columns
        Any additional columns to include in the index.
    check_duplicates
        Whether to check for duplicate entries in the index. If ``True``, an error
        will be raised if duplicates are found. This can significantly slow down the
        index generation for large datasets.
    sort
        Whether to sort the index by the index column. Sorting can cause OOM errors
        for large datasets, so it is disabled by default.
    max_rows_per_file
        The maximum number of rows per file when chunking the index.
        Defaults to 10 million rows per file.
    max_kilobytes_per_file
        The maximum number of kilobytes per file when chunking the index.
        Defaults to 10 GB.

    """

    object_path = get_dataset_path(object_path)
    index_path = pathlib.Path(index_path)

    if not object_path.exists():
        raise FileNotFoundError(f"No object catalogue found at {object_path!s}.")

    lf = polars.scan_parquet(object_path)

    columns = lf.collect_schema().names()
    for col in [index_column] + extra_columns:
        if col not in columns:
            raise ValueError(f"Column '{col}' not found in the object catalogue.")

    gb_cols = ["Norder", "Dir", "Npix", index_column]
    index_lf = lf.group_by(gb_cols)

    if check_duplicates:
        dupes = (
            index_lf.len(name="n").filter(polars.col.n > 1).collect(engine="streaming")
        )
        assert isinstance(dupes, polars.DataFrame)

        if dupes.height > 0:
            raise ValueError("Duplicate entries found in the index.")

    index_path.mkdir(parents=True, exist_ok=True)

    # Add the extra columns to the groupby aggregation.
    index_lf = index_lf.agg(polars.col(extra_columns).first())

    if sort:
        index_lf = index_lf.sort(index_column)

    index_lf.sink_parquet(
        polars.PartitionBy(
            base_path=index_path,
            file_path_provider=lambda args: str(
                index_path / f"part-{args.index_in_partition:06}.parquet"
            ),
            max_rows_per_file=max_rows_per_file,
            approximate_bytes_per_file=max_kilobytes_per_file * 1024,
        ),
        engine="streaming",
    )
