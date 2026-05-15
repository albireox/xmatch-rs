#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-05-15
# @Filename: hats.pyi
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

def hats_increase_order(
    path: str,
    order_0: int,
    order_1: int,
    pixels: list[int] | None,
    max_rows_per_pixel: int | None,
    max_kilobytes_per_pixel: int | None,
    n_workers: int | None,
    progress_file: str | None,
    keep_initial_order_directories: bool,
) -> None:
    """Repartitions a HATS dataset increasing the HEALPix order.

    Given a HATS dataset, reads the pixels at the specified ``order_0`` depth and,
    for a given list of pixels, repartitions the pixels to a higher HEALPix order
    ``order_1``. The resulting pixels are written to the original dataset directory
    under ``Noder=order_1``.

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
        ``Norder=<order_0>/Dir=?/Npixel=?`` directory). If :obj:`None`,
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

    """
    ...
