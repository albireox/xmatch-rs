#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-05-17
# @Filename: tools.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib
import re

import polars


__all__ = [
    "get_dataset_path",
    "get_dataset_df",
    "get_pixel_path",
    "get_dataset_nrows",
    "get_directory_size",
]


def get_dataset_path(base_path: pathlib.Path | str) -> pathlib.Path:
    """Returns the path to the dataset directory in a HATS dataset."""

    base_path = pathlib.Path(base_path)
    if not base_path.exists():
        raise ValueError(f"The provided path {base_path!s} does not exist.")
    elif not base_path.is_dir():
        raise ValueError(f"The provided path {base_path!s} is not a directory.")

    if base_path.name == "dataset":
        return base_path

    dataset_path = base_path / "dataset"
    if not dataset_path.exists():
        raise ValueError(f"Cannot find dataset directory at {dataset_path!s}.")
    elif not dataset_path.is_dir():
        raise ValueError(f"The path {dataset_path!s} is not a directory.")

    return dataset_path


def get_dataset_df(path: pathlib.Path, include_size: bool = True) -> polars.DataFrame:
    """Returns a DataFrame with the number of rows and size in Kb for each pixel."""

    dataset_path = get_dataset_path(path)

    lf = polars.scan_parquet(dataset_path)

    n_rows = (
        lf.group_by(["Norder", "Npix"])
        .agg(n_rows=polars.len())
        .collect(engine="streaming")
    )
    assert isinstance(n_rows, polars.DataFrame)

    if not include_size:
        return n_rows

    pixel_dirs = list(dataset_path.glob("Norder=*/Dir=*/Npix=*"))
    pixel_dirs = [d for d in pixel_dirs if d.is_dir()]

    sizes_kb = []
    for pixel_dir in pixel_dirs:
        size_kb = get_directory_size(pixel_dir)
        if match := re.match(r".*Norder=(\d+)/Dir=\d+/Npix=(\d+)/?$", str(pixel_dir)):
            order = int(match.group(1))
            npix = int(match.group(2))
            sizes_kb.append({"Norder": order, "Npix": npix, "size_kb": size_kb})

    df_sizes = polars.DataFrame(sizes_kb)

    return n_rows.join(df_sizes, on=["Norder", "Npix"], how="left")


def get_directory_size(path: pathlib.Path) -> int:
    """Returns the total size of a directory in kilobytes."""

    total_size = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                # entry.stat() provides size in bytes
                total_size += entry.stat().st_size

    return total_size // 1024  # Convert to kilobytes


def get_pixel_path(
    dataset_path: pathlib.Path,
    order: int,
    npix: int,
) -> pathlib.Path:
    """Returns the path to a pixel directory in a HATS dataset.

    No check is performed to verify that the pixel directory exists but the
    dataset path must exits.

    """

    dataset_path = get_dataset_path(dataset_path)

    dir_order = npix // 10_000 * 10_000
    pixel_dir = dataset_path / f"Norder={order}" / f"Dir={dir_order}" / f"Npix={npix}"

    # HATS supports the pixel path being .../Norder=<norder>/Dir=<dir>/Npix=<npix>/
    # or .../Norder=<order>/Dir=<dir>/Npix=<npix>.parquet. xmatch-rs always uses
    # the former but here we support both.
    if not pixel_dir.exists() and pixel_dir.with_suffix(".parquet").exists():
        pixel_dir = pixel_dir.with_suffix(".parquet")

    return pixel_dir


def get_dataset_nrows(
    dataset_path: pathlib.Path,
    orders: list[int] | None = None,
) -> int:
    """Returns the total number of rows in a HATS dataset."""

    dataset_path = get_dataset_path(dataset_path)

    lf = polars.scan_parquet(dataset_path)

    if orders is not None:
        lf = lf.filter(polars.col("Norder").is_in(orders))

    n_rows = lf.select(polars.len()).collect().item()  # type: ignore

    return n_rows
