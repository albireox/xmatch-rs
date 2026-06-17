#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-06-15
# @Filename: xmatch.pyi
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import polars

def init_logger(logger_name: str) -> None:
    """
    Initialises the logger bridge between Rust and Python.

    Parameters
    ----------
    logger_name
        The name of the logger to initialise.

    """
    ...

def get_max_catalogid(xmatch_path: str | pathlib.Path, run_id: int | None) -> int:
    """Returns the maximum catalog ID used in the cross-match process.

    Parameters
    ----------
    run_id
        The ID of the run to get the maximum catalog ID for. If `None`,
        returns the maximum catalog ID across all runs.

    Returns
    -------
    int
        The maximum catalog ID.

    """
    ...

def scan_pixel_from_dataset(
    hats_path: str | pathlib.Path,
    order: int,
    pixel: int,
    strict: bool,
) -> polars.LazyFrame | None:
    """Scans a HEALPix pixel from a HATS dataset.

    Given a HATS dataset and a ``pixel`` of a given ``order``, scans all the pixels in
    the dataset that contain the input pixel. Optionally, the dataset can be filtered
    to includel only the targets for the input pixel by using the ``strict`` parameter;
    otherwise the returned ``LazyFrame`` is a superset. If there are no pixels in the
    dataset that contain the input pixel, returns ``None``.


    Parameters
    ----------
    hats_path
        The path to the HATS dataset.
    order
        The order of the input HEALPix pixel.
    pixel
        The HEALPix pixel to scan for.
    strict
        Whether to filter the dataset to include only the targets for the input pixel.

    """
    ...

def run_phase_3(
    xmatch_path: str | pathlib.Path,
    version: str,
    epoch: float,
    pixelation_order: int,
    catalog_data: str,
    output_path: str | pathlib.Path,
    n_workers: int,
):
    """Runs phase 3 of the cross-match process.

    Parameters
    ----------
    version
        The version of the cross-match process.
    epoch
        The epoch of the catalogue.
    pixelation_order
        The order of the pixelation.
    catalog_data
        A JSON string representing the catalogue model.
    output_path
        The path where the output will be stored.
    n_workers
        The number of worker threads to use.

    """
    ...
