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

from typing import TYPE_CHECKING, overload

import polars

from xmatch_rs.schema import EpochFormat


if TYPE_CHECKING:
    from xmatch_rs.hats.hats import HATS


__all__ = [
    "get_dataset_path",
    "get_dataset_df",
    "get_pixel_path",
    "get_dataset_nrows",
    "get_directory_size",
    "calculate_max_proper_motion",
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

    n_rows = lf.select(polars.len()).collect().item()

    return n_rows


@overload
def dataset_has_duplicates(
    dataset_path: pathlib.Path,
    column_name: str,
    return_duplicates: bool = False,
) -> polars.DataFrame: ...


@overload
def dataset_has_duplicates(
    dataset_path: pathlib.Path,
    column_name: str,
    return_duplicates: bool = True,
) -> bool: ...


def dataset_has_duplicates(
    dataset_path: pathlib.Path,
    column_name: str,
    return_duplicates: bool = False,
) -> polars.DataFrame | bool:
    """Checks for duplicate rows in a HATS dataset."""

    dataset_path = get_dataset_path(dataset_path)

    lf = polars.scan_parquet(dataset_path)

    duplicates = lf.select(
        polars.col(column_name),
        polars.col(column_name).is_duplicated().alias("is_duplicated"),
    )

    duplicates = duplicates.filter(polars.col("is_duplicated")).select(column_name)

    if return_duplicates:
        return duplicates.collect(engine="streaming")

    n_duplicates = (
        duplicates.select(polars.col(column_name).len())
        .collect(engine="streaming")
        .item(0, 0)
    )

    return n_duplicates > 0


def calculate_max_proper_motion(
    hats: HATS,
    epoch: float | None = None,
    precise: bool = False,
    percentile: float = 100,
) -> float | None:
    """Returns the maximum proper motion in arcsec.

    Given a HATS dataset, returns the maximum expect proper motion for a certain
    ``epoch``. ``epoch`` must be provided in Julian years. If ``epoch=None`, the
    maximum proper motion per year is returned. If the catalogue does not provide
    proper motion information, returns `None`.

    If ``precise`` is True, performs the proper spherical trigonometry calculation
    of the total proper motion. Otherwise uses the approximation for small angles
    (i.e., the flat space Pythagorean theorem).

    ``percentile`` allows to define the percentile of the proper motion distribution to
    return. The default, ``percentile=100``, returns the maximum value.

    """

    if percentile < 0:
        raise ValueError("Percentile must be a non-negative value.")

    schema = hats.schema

    if (
        schema.model.pmra_column is None
        or schema.model.pmdec_column is None
        or schema.model.dec_column is None
    ):
        return None

    ref_epoch: float | None = None
    ref_epoch_column: str | None = None
    ref_epoch_format = schema.model.epoch_format

    if epoch:
        if schema.model.epoch is not None:
            ref_epoch = schema.model.epoch
            if ref_epoch_format == EpochFormat.JD:
                ref_epoch = (ref_epoch - 2451545) / 365.25 + 2000
        elif schema.model.epoch_column is not None:
            ref_epoch_column = schema.model.epoch_column
        else:
            return None

    # Get maximum proper motion per year.
    pmra_is_cosdec = schema.model.pmra_is_cosdec

    lf = hats.get_as_lazyframe()
    lf_pm = (
        lf.select(
            dec_rad=polars.col(schema.model.dec_column).radians(),
            pmra_rad=polars.col(schema.model.pmra_column).radians() / 1000 / 3600,
            pmdec_rad=polars.col(schema.model.pmdec_column).radians() / 1000 / 3600,
        )
        .drop_nulls()
        .drop_nans()
    )

    if pmra_is_cosdec is False:
        lf_pm = lf_pm.with_columns(
            pmra_rad=polars.col.pmra_rad * polars.col.dec_rad.cos()
        )

    if precise:
        lf_pm = lf_pm.with_columns(
            pm_total_deg=(
                polars.col.pmra_rad.abs().cos() * polars.col.pmdec_rad.abs().cos()
            )
            .arccos()
            .degrees()
            * 3600
        )
    else:
        lf_pm = lf_pm.with_columns(
            pm_total_deg=(polars.col.pmra_rad**2 + polars.col.pmdec_rad**2)
            .sqrt()
            .degrees()
            * 3600
        )

    if percentile >= 100.0:
        percentile = 100.0

    if epoch is None or ref_epoch_column is None:
        if epoch is not None:
            if ref_epoch is None:
                raise ValueError("Reference epoch is not available.")
            if epoch < ref_epoch:
                raise ValueError(
                    f"Epoch {epoch} is before catalogue reference epoch {ref_epoch}."
                )

        max_pm = lf_pm.select(
            polars.col.pm_total_deg.quantile(percentile / 100.0)
        ).collect(engine="streaming")
        max_pm = float(max_pm.item())

        if epoch is None:
            return max_pm
        else:
            assert ref_epoch is not None
            epoch_diff = epoch - ref_epoch
            return max_pm * epoch_diff

    assert ref_epoch_column is not None

    max_pm = (
        lf_pm.select(pm_epoch=polars.col(ref_epoch_column) * polars.col.pm_total_deg)
        .select(max_pm_epoch=polars.col.pm_epoch.quantile(percentile / 100.0))
        .collect(engine="streaming")
    )

    return float(max_pm.item())
