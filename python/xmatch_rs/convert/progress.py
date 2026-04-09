#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-04-08
# @Filename: progress.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import time

from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from xmatch_rs import console
from xmatch_rs.convert.models import ConvertToHATSModel


if TYPE_CHECKING:
    from xmatch_rs.convert.models import PhaseProgressModel


__all__: list[str] = ["monitor_progress", "get_phase", "get_n_pixels_written"]


def monitor_progress(
    output_path: pathlib.Path | str,
    phase_name: str,
    show_progress: bool = True,
    interval: float = 10,
    order: int | None = None,
    progress_bar_description: str | None = None,
) -> None:
    """Monitors the progress of the conversion and updates the progress model.

    This function can be run in a separate thread or process to monitor the
    progress of the conversion and update the progress model accordingly.

    Parameters
    ----------
    output_path
        The path where the conversion progress file is located.
    phase_name
        The name of the phase to monitor.
    show_progress
        Whether to show a progress bar.
    interval
        The interval in seconds to check for progress updates.
    order
        The HEALPix order to monitor for progress updates. If this argument
        is provided, the function will monitor the output directory
        and determine the number of tiles completed based on the HATS output
        structure. Otherwise the progress is determined by reading
        the ``_convert.yaml`` file.
    progress_bar_description
        An optional description to show in the progress bar.
    console
        An optional Rich console to use for the progress bar. If not provided,
        a new console will be created.

    """

    output_path = pathlib.Path(output_path)
    cmodel, phase = get_phase(output_path, phase_name)

    n_required = phase.tiles_required

    if show_progress:
        pbar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
            expand=True,
            auto_refresh=True,
        )
        pbar_task = pbar.add_task(
            progress_bar_description or f"Phase {phase.name}",
            total=n_required or 1,
        )
        pbar.start()
    else:
        pbar = None
        pbar_task = None

    while True:
        # Read phase again.
        cmodel, phase = get_phase(output_path, phase_name)

        if phase.complete:
            if pbar and pbar_task:
                pbar.update(pbar_task, completed=phase.tiles_required or 1)
                pbar.stop()
            break

        # If we are not monitoring the output directory,
        # we can just read the progress from the _convert.yaml file.
        if not order:
            n_processed = phase.tiles_processed
            if pbar is not None and pbar_task is not None and n_processed is not None:
                pbar.update(pbar_task, completed=n_processed, refresh=True)

        else:
            # Otherwise, we need to monitor the output directory and determine
            # the number of tiles completed based on the HATS output structure.
            n_processed = get_n_pixels_written(output_path, order)
            if pbar is not None and pbar_task is not None:
                pbar.update(pbar_task, completed=n_processed, refresh=True)

            cmodel.update_phase(phase_name, tiles_processed=n_processed)

        time.sleep(interval)


def get_phase(
    output_path: pathlib.Path | str,
    phase_name: str,
) -> tuple[ConvertToHATSModel, PhaseProgressModel]:
    """Returns the progress model for a given phase name."""

    model = ConvertToHATSModel.read(output_path)

    return model, model.get_phase(phase_name)


def get_n_pixels_written(output_path: pathlib.Path | str, order: int) -> int:
    """Returns the number of pixels written for a given HEALPix order."""

    order_path = pathlib.Path(output_path) / f"Norder={order}"

    return len(list(order_path.glob("Dir=*/Npix=*")))
