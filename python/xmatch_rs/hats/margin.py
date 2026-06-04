#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-06-02
# @Filename: margin.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import json
import multiprocessing
import pathlib
import shutil
import time

from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from sdsstools import Timer

import xmatch_rs._xmatch_rs_lib.hats as _rs_hats
from xmatch_rs import console
from xmatch_rs.tools import get_xmatch_rs_logger


if TYPE_CHECKING:
    from sdsstools.logger import SDSSLogger


def create_margin_cache(
    collection_path: str | pathlib.Path,
    margin_name: str,
    buffer: float,
    overwrite: bool = False,
    n_workers: int | None = None,
    show_progress: bool = True,
    log: SDSSLogger | None = None,
    quiet: bool = False,
):
    """ "Creates a margin cache for a HATS collection.

    Parameters
    ----------
    collection_path
        The path to the HATS collection.
    margin_name
        The name of the margin cache to create. This is used to identify the margin
        cache and should be unique within the collection.
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
    show_progress
        If ``True``, displays a progress bar during the operation.
    log
        An optional logger to use for logging the conversion process.
    quiet
        If ``True``, suppresses logging output. Defaults to ``False``. Ignored if
        a logger is provided via the ``log`` parameter.

    """

    # Get logger
    log = log or get_xmatch_rs_logger(log_level=40 if quiet else 20)

    collection_path = pathlib.Path(collection_path)
    margin_path = collection_path / margin_name

    if margin_path.exists():
        if overwrite:
            shutil.rmtree(margin_path)
        else:
            raise FileExistsError(
                f"The margin cache directory {margin_path!s} already exists."
            )

    log.info(
        f"Creating margin cache {margin_name!r} with buffer {buffer} arcsec "
        f"for {collection_path!s}."
    )

    progress_file = margin_path / "_progress.json"

    _monitor_process = multiprocessing.Process(
        target=_monitor_progress,
        args=(progress_file if show_progress else None, margin_name),
        daemon=True,
    )
    _monitor_process.start()

    # Call the Rust function to create the margin cache.
    try:
        with Timer() as timer:
            _rs_hats.create_margin_cache(
                collection_path=collection_path,
                margin_name=margin_name,
                buffer=buffer,
                overwrite=overwrite,
                n_workers=n_workers,
                progress_file=progress_file if show_progress else None,
            )
    except Exception:
        log.error("Error during margin cache creation.")
        _monitor_process.terminate()
        raise
    finally:
        progress_file.unlink(missing_ok=True)

    # Make sure the monitoring process finished.
    _monitor_process.join()

    log.info(f"Margin cache {margin_name!r} created in {timer.duration:,.2f} s.")


def _monitor_progress(progress_file: pathlib.Path | None, margin_name: str) -> None:
    """Monitors the progress of the re-partitioning process and logs it."""

    if progress_file is None:
        return

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

    while True:
        if progress_file.exists():
            with open(progress_file, "r") as f:
                try:
                    progress_data = json.load(f)
                except json.JSONDecodeError:
                    # In case we are reading the file as it's being written
                    # and it's not valid JSON yet, we just wait and try again.
                    time.sleep(2)
                    continue

                to_process: int = progress_data.get("n_pixels_to_process", 0)
                completed: int = progress_data.get("n_pixels_completed", 0)
                done = progress_data.get("complete", False)

            if pbar_task is None:
                pbar_task = pbar.add_task(
                    f"{margin_name!s} ...",
                    total=to_process,
                )
                pbar.start()

            if done:
                pbar.update(pbar_task, completed=to_process, total=to_process)
                break

            pbar.update(pbar_task, completed=completed, total=to_process)

        time.sleep(2)  # Sleep for a bit before checking again

    if pbar_task is not None:
        pbar.stop()

    return
