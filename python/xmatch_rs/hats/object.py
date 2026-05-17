#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-04-14
# @Filename: object.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import json
import multiprocessing
import os
import pathlib
import re
import shutil
import time

from typing import TYPE_CHECKING, Literal

import polars
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from sdsstools import Timer, read_yaml_file

from xmatch_rs import console
from xmatch_rs._xmatch_rs_lib.hats import hats_repartition
from xmatch_rs.convert import convert_csv_to_parquet, convert_to_parquet
from xmatch_rs.hats.models import ConvertToHATSModel, PhaseModel
from xmatch_rs.hats.providers import (
    get_hats_file_provider,
)
from xmatch_rs.schema import Schema
from xmatch_rs.tools import get_xmatch_rs_logger, parse_path


if TYPE_CHECKING:
    from sdsstools.logger import SDSSLogger

    from xmatch_rs.types import APPLY_CALLBACK_T, DATA_PATH_T


__all__: list[str] = [
    "convert_to_hats_object_catalogue",
    "get_dataset_nrows",
    "check_completion",
    "get_dataset_completion_df",
    "get_directory_size",
    "get_dataset_path",
]


def convert_to_hats_object_catalogue(
    data_path: DATA_PATH_T,
    collection_path: str | pathlib.Path,
    schema: Schema | None,
    format: Literal["csv", "parquet"] | None = None,
    initial_order: int = 2,
    reference_order: int = 29,
    max_order: int | None = None,
    max_rows_per_pixel: int | None = 1_000_000,
    max_kilobytes_per_pixel: int | None = None,
    apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None = None,
    n_workers: int | None = None,
    dynamic_planning: bool = True,
    show_progress: bool = True,
    log: SDSSLogger | None = None,
    quiet: bool = False,
    overwrite: bool = False,
) -> ConvertToHATSModel:
    """Converts a dataset to a HATS object catalogue.

    This function creates the structure of a HATS collection but does not populate
    the metadata files for the collection or catalogue. Use
    :obj:`~xmatch_rs.hats.HATS.convert_to_hats` for a higher-level function
    that also creates the metadata files.

    Parameters
    ----------
    data_path
        The path(s) to the dataset to be converted to HATS.
    collection_path
        The path where the output HATS object catalogue will be created. The resulting
        directory structure will be ``<collection_path>/object/dataset/...``
        along with the necessary metadata files.
    schema
        The schema of the input dataset. Must be a :obj:`~xmatch_rs.schema.Schema`
        object.
    format
        The format of the output files. If not provided, it will be inferred from
        the files in ``data_path``.
    initial_order
        The initial HEALPix order to use for partitioning the data. Defaults to 6.
    reference_order
        The reference HEALPix order. A column ``_healpix_{reference_order}``
        will be added to the dataset. This should be a relatively high order,
        ideally higher than any reasonable spatial resolution that the dataset
        (or any other dataset with which it will be cross-matched) can achieve.
        Defaults to 29, which corresponds to a resolution of ~0.000393 arcseconds.
    max_order
        The maximum HEALPix order to use for partitioning the data. If not
        provided, the tiling process will continue until all pixels contain
        fewer than ``max_rows_per_pixel`` rows.
    max_rows_per_pixel
        The maximum number of rows per pixel. If set to ``None`` (and
        ``max_kilobytes_per_pixel`` is also ``None``), no adaptive
        tiling will be performed and the data will be partitioned using the
        ``initial_order``.
    max_kilobytes_per_pixel
        The maximum number of kilobytes per pixel. If set to ``None`` (and
        ``max_rows_per_pixel`` is also ``None``), no adaptive
        tiling will be performed and the data will be partitioned using the
        ``initial_order``.
    n_workers
        The number of worker processes to use for the conversion. If not provided,
        uses recommended defaults based on the system's resources. ``n_workers=1``
        will disable multiprocessing and threading in most cases and processing
        will be done sequentially in a single process.
    dynamic_planning
        If ``True``, the code will dynamically decide whether to add phases to
        the conversion process to ensure that ``max_rows_per_pixel`` and
        ``max_kilobytes_per_pixel`` are not exceeded. If ``False``, a conversion
        YAML file is expected to be present in the output directory and the conversion
        process will follow the phases defined in that file.
    apply_callback
        A callback function or list of callback functions to apply to the data
        before saving. Each callback should take a Polars LazyFrame as input
        and return a modified LazyFrame.
    show_progress
        Track conversion progress and show a progress bar. Progress tracking
        can be inaccurate. For the initial pixelation, in particular, the estimate
        will only be accurate if the input data is approximately all-sky.
    log
        An optional logger to use for logging the conversion process.
    quiet
        If ``True``, suppresses logging output. Defaults to ``False``. Ignored if
        a logger is provided via the ``log`` parameter.
    overwrite
        If ``True``, allows overwriting existing files in the output directory.
        defaults to ``False``.

    """

    # Get logger
    log = log or get_xmatch_rs_logger(log_level=40 if quiet else 20)

    # Make sure data_path is a list of paths.
    if isinstance(data_path, (str, pathlib.Path)):
        data_path = [str(data_path)]
    else:
        data_path = [str(p) for p in data_path]

    # Check n_workers
    if n_workers is not None and n_workers < 1:
        log.warning(
            "n_workers must be a positive integer or None. "
            "Defaulting to n_workers=None."
        )
        n_workers = None

    # Create the object and dataset directories.
    collection_path = pathlib.Path(collection_path)

    object_path = collection_path / "object"
    object_path.mkdir(parents=True, exist_ok=True)

    dataset_path = object_path / "dataset"
    if dataset_path.exists() and not overwrite:
        raise FileExistsError(
            f"The dataset directory {dataset_path!s} already exists. "
            "Set overwrite=True to allow overwriting.",
        )

    dataset_path.mkdir(parents=True, exist_ok=True)

    # Try to load the _convert.yaml file or create a new instance.
    if (object_path / "_convert.yaml").exists():
        log.info(f"Loading _convert.yaml from {object_path!s}.")
        convert_dict = read_yaml_file(object_path / "_convert.yaml", return_class=dict)
        convert_model = ConvertToHATSModel.model_validate(convert_dict)
        if schema is not None:
            convert_model.catalogue_schema = schema.schema
    else:
        if not dynamic_planning:
            raise FileNotFoundError(
                f"No _convert.yaml file found in {object_path!s} and "
                "dynamic_planning is False. Please provide a conversion model "
                "or set dynamic_planning to True.",
            )

        if schema is None:
            raise ValueError(
                "No schema provided and no _convert.yaml file found in the output "
                "directory. Please provide a schema for the input data.",
            )

        # Build the conversion model and the initial phase and
        # write the _convert.yaml file to the output directory.
        healpix_min_order_model = PhaseModel(
            name=f"healpix_{initial_order}_partitioning",
            type="healpix_partitioning",
            description=f"Pixelation to order {initial_order}.",
            healpix_order_0=None,
            healpix_order_1=initial_order,
        )

        convert_model = ConvertToHATSModel(
            catalogue_schema=schema.schema,
            object_path=object_path,
            complete=False,
            reference_order=reference_order,
            phases=[healpix_min_order_model],
        )
        convert_model.write()

    # Check if conversion is complete.
    if convert_model.complete:
        log.info("Conversion to HATS object catalogue is complete. Nothing left to do.")
        return convert_model

    while not convert_model.complete:
        # Get next phase to run.
        phase = convert_model.get_next_phase()

        if not phase:
            log.info("No more phases to run. Checking if conversion is complete.")
            result = check_completion(
                dataset_path,
                max_rows_per_pixel=max_rows_per_pixel,
                max_kilobytes_per_pixel=max_kilobytes_per_pixel,
            )

            if result:
                log.info("Conversion to HATS object catalogue is complete.")
                convert_model.complete = True
                convert_model.write()
                break
            else:
                if not dynamic_planning:
                    log.warning(
                        "Conversion is not complete but dynamic_planning "
                        "is False. Stopping."
                    )
                    break

                log.info("Conversion is not complete. Adding re-partitioning phase.")
                phase = add_repartitioning_phase(
                    convert_model,
                    max_rows_per_pixel=max_rows_per_pixel,
                    max_kilobytes_per_pixel=max_kilobytes_per_pixel,
                )

        if phase.type == "healpix_partitioning":
            _run_hats_healpix_phase(
                data_path=data_path,
                model=convert_model,
                phase_name=phase.name,
                format=format,
                apply_callback=apply_callback,
                log=log,
                show_progress=show_progress,
                n_workers=n_workers,
            )
        else:
            raise ValueError(f"Unknown phase type {phase.type!r}.")

        if not phase.complete:
            raise RuntimeError(f"Phase {phase.name!r} did not complete successfully.")

        log.info("Checking if HATS conversion is complete.")
        complete = check_completion(
            dataset_path,
            max_rows_per_pixel=max_rows_per_pixel,
            max_kilobytes_per_pixel=max_kilobytes_per_pixel,
        )

        if complete:
            log.info("Conversion to HATS object catalogue is complete.")
            convert_model.complete = True
            convert_model.write()
            break

        if not dynamic_planning:
            continue

        if (
            phase.healpix_order_1 is not None
            and max_order is not None
            and phase.healpix_order_1 >= max_order
        ):
            log.info(
                f"Reached maximum HEALPix order {max_order}. "
                "Marking conversion as complete.",
            )
            convert_model.complete = True
            convert_model.write()
            break

        # Add the next phase for increasing the HEALPix order.
        if phase.healpix_order_1 is None:
            raise ValueError(
                "Found null healpix_order_1. Cannot continue partitioning."
            )

        log.info("Adding next phase for increasing HEALPix order.")
        add_repartitioning_phase(
            convert_model,
            order_0=phase.healpix_order_1,
            order_1=phase.healpix_order_1 + 1,
            max_rows_per_pixel=max_rows_per_pixel,
            max_kilobytes_per_pixel=max_kilobytes_per_pixel,
        )

    return convert_model


def _run_hats_healpix_phase(
    data_path: DATA_PATH_T,
    model: ConvertToHATSModel,
    phase_name: str,
    log: SDSSLogger,
    format: Literal["csv", "parquet"] | None = None,
    apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None = None,
    show_progress: bool = True,
    n_workers: int | None = None,
) -> None:
    """Runs the first phase of the HATS conversion: convert to a fixed tiling order."""

    schema = model.get_schema()
    phase = model.get_phase(phase_name)

    if phase.complete:
        log.info(f"Phase {phase_name!r} already completed. Skipping.")
        return

    order_0 = phase.healpix_order_0
    order_1 = phase.healpix_order_1

    if order_0 is None:
        # Determine the file format if not provided.
        if format is None:
            files = parse_path(data_path)
            sample_file = files[0]
            if re.match(r".*\.csv\.?[gz|bz2]?$", sample_file.name):
                log.info("Inferred file format as CSV based on file extension.")
                format = "csv"
            elif re.match(r".*\.parquet$", sample_file.name):
                log.info("Inferred file format as Parquet based on file extension.")
                format = "parquet"
            else:
                raise ValueError(
                    "Could not infer file format from file extension. "
                    "Please specify the format explicitly.",
                )

        assert order_1 is not None, "healpix_order_1 must be set for the initial phase."

        # This is the initial partition to HATS format.
        _run_hats_min_order_phase(
            data_path=data_path,
            model=model,
            schema=schema,
            phase_name=phase_name,
            format=format,
            order_1=order_1,
            apply_callback=apply_callback,
            log=log,
            show_progress=show_progress,
            n_workers=n_workers,
        )
    else:
        # Re-partition from order_0 to order_1.
        assert order_1 is not None, "healpix_order_1 must be set for re-partitioning."
        _run_hats_repartition(
            model=model,
            phase_name=phase_name,
            order_0=order_0,
            order_1=order_1,
            log=log,
            show_progress=show_progress,
            n_workers=n_workers,
        )


def _run_hats_min_order_phase(
    data_path: DATA_PATH_T,
    model: ConvertToHATSModel,
    schema: Schema,
    phase_name: str,
    order_1: int,
    log: SDSSLogger,
    format: Literal["csv", "parquet"] | None = None,
    apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None = None,
    show_progress: bool = True,
    n_workers: int | None = None,
) -> None:
    """Runs the initial HEALPix partitioning phase of the HATS conversion."""

    # Get the file provider for HATS files.
    hats_file_provider = get_hats_file_provider(
        order_1,
        column_name=f"_healpix_{order_1}",
        use_uuid=True,
    )

    # The output path for the Norder hive files, following the HATS convention.
    output_path = model.object_path / "dataset"
    output_path.mkdir(parents=True, exist_ok=True)

    with Timer() as timer:
        if format == "csv":
            log.info("Converting CSV files to HATS.")
            log.info(f"Performing initial convertion to order {order_1}.")

            convert_csv_to_parquet(
                csv_files=data_path,
                schema=schema,
                output_path=output_path,
                mode="partition",
                partition_process_mode="single",
                partition_key=f"_healpix_{order_1}",
                file_path_provider=hats_file_provider,
                partition_include_key=False,
                parallelise=True if n_workers is not None and n_workers > 1 else False,
                n_workers=n_workers,
                healpix_orders=[model.reference_order, order_1],
                healpix_column_name="_healpix_{order}",
                apply_callback=apply_callback,
                show_progress=show_progress,
                log=log,
            )

        elif format == "parquet":
            log.info("Converting Parquet files to HATS.")
            log.info(f"Performing initial convertion to order {order_1}.")

            if isinstance(data_path, (str, pathlib.Path)):
                data_path = [data_path]

            lf = polars.scan_parquet([pathlib.Path(p) for p in data_path])

            convert_to_parquet(
                lf,
                schema=schema,
                output_paths=output_path,
                mode="partition",
                partition_process_mode="single",
                partition_key=f"_healpix_{order_1}",
                file_path_provider=hats_file_provider,
                partition_include_key=False,
                parallelise=True,
                n_workers=n_workers,
                healpix_orders=[model.reference_order, order_1],
                healpix_column_name="_healpix_{order}",
                apply_callback=apply_callback,
                log=log,
                show_progress=show_progress,
            )

        else:
            raise ValueError(
                f"Invalid format {format!r}. Currently supported formats "
                "are 'csv' and 'parquet'.",
            )

        log.info(f"Phase {phase_name!r} completed in {timer.elapsed:,.1f} seconds.")

        model.update_phase(phase_name, complete=True)


def _run_hats_repartition(
    model: ConvertToHATSModel,
    phase_name: str,
    order_0: int,
    order_1: int,
    log: SDSSLogger | None = None,
    show_progress: bool = False,
    n_workers: int | None = None,
) -> list[int]:
    """Runs a re-partitioning phase of the HATS conversion.

    This phase will select pixels in the ``Norder=order_0`` directories that
    have more than ``max_rows_per_pixel`` rows or use more than
    ``max_kilobytes_per_pixel`` kilobytes, and split them into smaller pixels
    at HEALPix order ``order_1``. The original ``Norder=order_0`` directories are
    deleted after the operation.

    """

    log = log or get_xmatch_rs_logger()

    phase = model.get_phase(phase_name)

    if phase.complete:
        log.info(f"Phase {phase_name!r} already completed. Skipping.")
        return []

    dataset_path = get_dataset_path(model.object_path)
    order_0_path = dataset_path / f"Norder={order_0}"
    if not order_0_path.exists():
        raise ValueError(f"Cannot find directory Norder={order_0} at {order_0_path!s}.")

    log.info(
        f"Starting re-partitioning from order {order_0} to order {order_1} "
        f"for phase {phase_name!r}.",
    )

    t0 = time.time()

    # Get the number of rows before partitioning for validation.
    log.debug(f"Getting number of rows in order {order_0} before re-partitioning.")
    n_rows_order_0 = get_dataset_nrows(dataset_path, orders=[order_0])
    log.debug(f"Number of rows in order {order_0}: {n_rows_order_0:,}.")

    # The re-partitioning is fully implemented in Rust. We run the progress monitoring
    # in a different process (not a differet thread since PyO3 blocks the GIL).
    progress_file = model.object_path / f"{phase_name}_progress.json"

    _monitor_process = multiprocessing.Process(
        target=_monitor_progress,
        args=(progress_file if show_progress else None, log),
        daemon=True,
    )
    _monitor_process.start()

    # Now we call the Rust code.
    try:
        # List of pixels that were re-partitioned.
        pixels: list[int] = hats_repartition(
            path=str(dataset_path),
            order_0=order_0,
            order_1=order_1,
            pixels=None,
            max_rows_per_pixel=phase.max_rows_per_pixel,
            max_kilobytes_per_pixel=phase.max_kilobytes_per_pixel,
            n_workers=n_workers,
            progress_file=str(progress_file) if show_progress else None,
            keep_initial_order_directories=True,  # We'll delete them ourselves later
        )
    except Exception:
        log.error(f"Error during re-partitioning to order {order_1}.")
        raise

    # Make sure the monitoring process finished.
    _monitor_process.join()

    # Check the Norder=order_0 directories.
    log.info(f"Validating re-partitioning to order {order_1}.")
    check_completion(
        dataset_path=dataset_path,
        order_0=order_0,
        order_1=order_1,
        max_rows_per_pixel=phase.max_rows_per_pixel,
        max_kilobytes_per_pixel=phase.max_kilobytes_per_pixel,
        skip_pixels=pixels,
        expected_n_rows=n_rows_order_0,
        raise_=True,
    )

    # Delete the original order_0 directories.
    elapsed = time.time() - t0
    log.info(f"Re-partitioning to order {order_1} complete in {elapsed:,.1f} seconds.")

    if phase.keep_lower_order_directories is False:
        log.info(f"Removing order {order_0} directories.")
        for pixel in pixels:
            pixel_dir = get_pixel_path(dataset_path, order_0, pixel, check=False)
            if pixel_dir.exists() and pixel_dir.is_dir():
                shutil.rmtree(pixel_dir)

    # Mark phase as complete.
    log.info(f"Phase {phase_name!r} completed.")
    model.update_phase(phase_name, complete=True)

    # Remove progress file.
    if progress_file and progress_file.exists():
        log.debug("Removing progress file.")
        progress_file.unlink()

    return pixels


def _monitor_progress(progress_file: pathlib.Path | None, log: SDSSLogger) -> None:
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

    message_shown: bool = False

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

                order_0: int = progress_data.get("order_0", -1)
                order_1: int = progress_data.get("order_1", -1)
                to_process: list[int] = progress_data.get("pixels_to_process", [])
                completed: list[int] = progress_data.get("pixels_completed", [])
                done = progress_data.get("complete", False)

                if not message_shown:
                    n_pixels = len(to_process)
                    log.info(
                        f"Re-partitioning {n_pixels} pixels from order {order_0} "
                        f"to order {order_1}: {to_process}"
                    )
                    message_shown = True

            if pbar_task is None:
                pbar_task = pbar.add_task(
                    "Re-partitioning pixels",
                    total=len(to_process),
                )
                pbar.start()

            if done:
                pbar.update(pbar_task, completed=len(to_process), total=len(to_process))
                break

            pbar.update(pbar_task, completed=len(completed), total=len(to_process))

        time.sleep(2)  # Sleep for a bit before checking again

    if pbar_task is not None:
        pbar.stop()

    return


def get_dataset_path(base_path: pathlib.Path) -> pathlib.Path:
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


def get_dataset_completion_df(
    path: pathlib.Path,
    include_size: bool = True,
) -> polars.DataFrame:
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


def check_completion(
    dataset_path: pathlib.Path,
    order_0: int | None = None,
    order_1: int | None = None,
    expected_n_rows: int | None = None,
    max_rows_per_pixel: int | None = None,
    max_kilobytes_per_pixel: int | None = None,
    skip_pixels: list[int] | None = None,
    raise_: bool = False,
) -> bool:
    """Checks the pixels in a HATS dataset for completion.

    This function checks the pixels in the provided path for whether they contain
    more than ``max_rows_per_pixel`` rows or more than
    ``max_kilobytes_per_pixel`` kilobytes. It can also check if the total number of
    rows in the pixels of order_0 and order_1 matches the expected number of rows.

    Returns
    -------
    bool
        ``True`` if all pixels meet the criteria, ``False`` otherwise.

    """

    if max_kilobytes_per_pixel is None and max_rows_per_pixel is None:
        return True

    dataset_path = get_dataset_path(dataset_path)

    data = get_dataset_completion_df(
        dataset_path,
        include_size=max_kilobytes_per_pixel is not None,
    )

    if expected_n_rows is not None:
        # Number of rows in pixels of order_1.
        n_rows_order_1 = (
            data.filter(polars.col.Norder == order_1)
            .select(polars.sum("n_rows"))
            .item()
        )

        # It's a bit more complicated for order_0 because we want to exclude pixels
        # that were re-partitioned to order_1.
        n_rows_order_0 = (
            data.filter(polars.col.Norder == order_0)
            .filter(
                polars.col.Npix.is_in(skip_pixels).not_()
                if skip_pixels
                else polars.lit(True)
            )
            .select(polars.sum("n_rows"))
            .item()
        )

        n_rows_total = n_rows_order_0 + n_rows_order_1
        if n_rows_total != expected_n_rows:
            if raise_:
                raise RuntimeError(
                    f"Re-partitioning to order {order_1} did not complete "
                    f"successfully. The number of rows after re-partitioning does not "
                    f"match the expected number of rows. Expected {expected_n_rows:,} "
                    f"rows but found {n_rows_total:,}. The original pixel directories "
                    f"have not been deleted."
                )
            return False

    if order_0 is not None:
        data = data.filter(polars.col.Norder == order_0)

    if skip_pixels is not None:
        data = data.filter(polars.col.Npix.is_in(skip_pixels).not_())

    if max_rows_per_pixel is not None:
        if data.filter(polars.col.n_rows > max_rows_per_pixel).height > 0:
            if raise_:
                raise RuntimeError(
                    f"Re-partitioning to order {order_1} did not complete "
                    f"successfully. Some pixels in order {order_0} contain more than "
                    f"{max_rows_per_pixel:,} rows. The original pixel directories have "
                    f"not been deleted."
                )
            return False

    if max_kilobytes_per_pixel is not None:
        if data.filter(polars.col.size_kb > max_kilobytes_per_pixel).height > 0:
            if raise_:
                raise RuntimeError(
                    f"Re-partitioning to order {order_1} did not complete "
                    f"successfully. Some pixels in order {order_0} contain more than "
                    f"{max_kilobytes_per_pixel:,} kilobytes. The original pixel "
                    f"directories have not been deleted."
                )
            return False

    return True


def get_pixel_path(
    dataset_path: pathlib.Path,
    order: int,
    npix: int,
    check: bool = False,
) -> pathlib.Path:
    """Returns the path to a pixel directory in a HATS dataset."""

    dataset_path = get_dataset_path(dataset_path)

    dir_order = npix // 10_000
    pixel_dir = dataset_path / f"Norder={order}" / f"Dir={dir_order}" / f"Npix={npix}"

    if check:
        if not pixel_dir.exists():
            raise ValueError(f"Pixel directory {pixel_dir!s} does not exist.")
        elif not pixel_dir.is_dir():
            raise ValueError(f"Pixel path {pixel_dir!s} is not a directory.")

    return pixel_dir


def add_repartitioning_phase(
    model: ConvertToHATSModel,
    order_0: int | None = None,
    order_1: int | None = None,
    max_rows_per_pixel: int | None = None,
    max_kilobytes_per_pixel: int | None = None,
) -> PhaseModel:
    """Adds a re-partitioning phase to the conversion model."""

    if order_0 is not None and order_1 is None:
        order_1 = order_0 + 1
    elif order_0 is None and order_1 is None:
        max_order = 0
        for phase in model.phases:
            if phase.complete and phase.healpix_order_1 is not None:
                if phase.healpix_order_1 > max_order:
                    max_order = phase.healpix_order_1
        order_0 = max_order
        order_1 = max_order + 1

    phase = PhaseModel(
        name=f"healpix_{order_1}_partitioning",
        type="healpix_partitioning",
        description=(f"Re-pixelation to order {order_1}."),
        healpix_order_0=order_0,
        healpix_order_1=order_1,
        max_rows_per_pixel=max_rows_per_pixel,
        max_kilobytes_per_pixel=max_kilobytes_per_pixel,
    )

    model.add_phase(phase)

    return phase


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
