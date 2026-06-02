#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-05-15
# @Filename: hats.pyi
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

def repartition(
    path: str | pathlib.Path,
    order_0: int,
    order_1: int,
    pixels: list[int] | None,
    max_rows_per_pixel: int | None,
    max_kilobytes_per_pixel: int | None,
    n_workers: int | None,
    progress_file: str | pathlib.Path | None,
    keep_initial_order_directories: bool,
) -> list[int]:
    """Repartitions a HATS dataset increasing the HEALPix order.

    Given a HATS dataset, reads the pixels at the specified ``order_0`` depth and,
    for a given list of pixels, repartitions the pixels to a higher HEALPix order
    ``order_1``. The resulting pixels are written to the original dataset directory
    under ``Norder=order_1``.

    Parameters
    ----------
    path
        The path to the HATS dataset.
    order_0
        The HEALPix order of the pixels to re-partition.
    order_1
        The HEALPix order of the resulting pixels.
    pixels
        The list of pixels to re-partition. If :obj:`None`, pixels that have larger
        than ``max_rows_per_pixel`` or ``max_kilobytes_per_pixel`` are selected.
    max_rows_per_pixel
        The maximum number of rows per pixel. If :obj:`None`, no limit is applied.
    max_kilobytes_per_pixel
        The maximum number of kilobytes per pixel (i.e., of the
        ``Norder=<order_0>/Dir=?/Npix=?`` directory). If :obj:`None`,
        no limit is applied.
    n_workers
        The number of workers to use. If :obj:`None`, uses half the available CPU cores.
    progress_file
        The path to a JSON file where the progress of the operation is stored. If
        :obj:`None`, no progress data is saved. The written file has JSON format
        and includes keys ``pixels_to_process`` and ``pixels_completed``, and can
        be used to monitor and display the progress of the operation.
    keep_initial_order_directories
        If ``True``, the original ``Norder=order_0`` directories are not deleted
        after the operation.

    Returns
    -------
    list[int]
        The list of pixels that were re-partitioned.

    """
    ...

def rechunk(
    path: str | pathlib.Path,
    is_dataset: bool,
    max_rows_per_file: int | None,
    approximate_bytes_per_file: int | None,
) -> None:
    """Rechunks a HATS dataset.

    Given a HATS dataset, reads the pixels at the specified depth and rechunks them
    into smaller files. The resulting pixels are written to the original dataset
    directory under ``Norder=order``.

    Parameters
    ----------
    path
        The path to the HATS dataset.
    is_dataset
        Whether the provided path is the dataset directory (i.e., the one containing
        the ``Norder=`` subdirectories). If ``True``, will rechunk any
        ``Norder=*/Dir=*/Npix=*`` directory under the provided path. If ``False``,
        will rechunk the provided path directly.
    max_rows_per_file
        The maximum number of rows per file. If :obj:`None`, defaults to 1 million
        rows per file.
    approximate_bytes_per_file
        The approximate number of bytes per file. If :obj:`None`, no limit is applied.

    """
    ...

def create_index(
    collection_path: str | pathlib.Path,
    index_name: str,
    index_column: str,
    extra_columns: list[str] | None,
    mode: str | None,
    check_duplicates: bool | None,
    max_rows_per_file: int | None,
    max_kilobytes_per_file: int | None,
) -> None:
    """Creates a HATS index for a given collection.

    Parameters
    ----------
    collection_path
        The path to the collection for which to create the index.
    index_name
        The name of the index to create. This will be used as the name of the
        directory where the index is stored.
    index_column
        The name of the column to use as the index key. This column must exist in
        the dataset and should contain unique values for each row.
    extra_columns
        A list of additional columns to include in the index. These columns will be
        included in the resulting index files and can be used for filtering or
        other operations when querying the index.
    mode
        The mode for creating the index. Can be "sequential" or "full".
    check_duplicates
        Whether to check for duplicate rows in the index. Only applies to "full" mode.
    max_rows_per_file
        The maximum number of rows per file.
    max_kilobytes_per_file
        The maximum size of each file in kilobytes.

    """
    ...

def create_margin_cache(
    collection_path: str | pathlib.Path,
    margin_name: str,
    buffer: float,
    overwrite: bool,
    n_workers: int | None,
    progress_file: str | pathlib.Path | None,
):
    """Creates a margin cache in a HATS collection.

    Parameters
    ----------
    collection_path
        The path to the HATS collection.
    margin_name
        The name of the margin cache to create. The margin cache dataset will be
        created under the directory ``<collection_path>/<margin_name>/dataset``.
    buffer
        The buffer value to use for the margin cache, in arcseconds. This is the
        maximum distance around each HATS partition pixel for which additional
        sources will be included in the margin cache.
    overwrite
        If ``True``, deletes the existing margin cache if it already exists.
        If ``False``, raises an error.
    n_workers
        The number of workers to use for the operation. If :obj:`None`, uses all
        the available CPU cores.
    progress_file
        The path to a JSON file where the progress of the operation is stored. If
        :obj:`None`, no progress data is saved. The written file has JSON format
        and includes keys ``n_pixels_to_process``, ``n_pixels_completed``, and
        ``complete`, and can be used to monitor and display the progress of the
        operation.

    """
    ...
