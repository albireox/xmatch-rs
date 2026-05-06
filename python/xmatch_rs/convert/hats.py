#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-04-14
# @Filename: hats.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import re

from typing import TYPE_CHECKING, Any, Literal

import polars

from sdsstools import Timer, read_yaml_file

from xmatch_rs.convert import (
    PhaseModel,
    convert_csv_to_parquet,
    convert_to_parquet,
    get_hats_file_provider,
)
from xmatch_rs.convert.models import ConvertToHATSModel
from xmatch_rs.schema import Schema
from xmatch_rs.tools import get_xmatch_rs_logger, parse_path
from xmatch_rs.types import LOW_MEMORY_STRATEGY_OPTIONS_T


if TYPE_CHECKING:
    from sdsstools.logger import SDSSLogger

    from xmatch_rs.types import (
        APPLY_CALLBACK_T,
        DATA_PATH_T,
        LOW_MEMORY_STRATEGY_T,
    )


__all__: list[str] = ["convert_to_hats"]


_LOW_MEMORY_DEFAULTS: dict[LOW_MEMORY_STRATEGY_OPTIONS_T, Any] = {
    "data_page_size": 0.1 * 1024 * 1024,
    "streaming_chunk_size": 500,
}


def get_low_memory_strategy_value(
    input: list[LOW_MEMORY_STRATEGY_OPTIONS_T]
    | dict[LOW_MEMORY_STRATEGY_OPTIONS_T, Any],
    strategy: LOW_MEMORY_STRATEGY_OPTIONS_T,
) -> int:
    """Returns the value for a low memory strategy, given the input and defaults."""

    if isinstance(input, dict):
        value = input.get(strategy, _LOW_MEMORY_DEFAULTS[strategy])
        if value is None:
            return _LOW_MEMORY_DEFAULTS[strategy]
        return value

    return _LOW_MEMORY_DEFAULTS[strategy]


def convert_to_hats(
    data_path: DATA_PATH_T,
    output_path: str | pathlib.Path,
    schema: Schema | str | pathlib.Path | dict | None = None,
    format: Literal["csv", "parquet"] | None = None,
    hats_initial_order: int = 6,
    hats_max_order: int | None = None,
    hats_max_rows_per_pixel: int | None = None,
    partition_max_rows_per_file: int | None = None,
    partition_approximate_bytes_per_file: int | Literal["auto"] | None = None,
    apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None = None,
    low_memory_strategy: LOW_MEMORY_STRATEGY_T | None = None,
    convert_kwargs: dict | None = None,
    preserve_progress: bool = True,
    show_progress: bool = False,
    log: SDSSLogger | None = None,
    quiet: bool = False,
) -> None:
    """Converts a dataset to the HATS format.

    Parameters
    ----------
    data_path
        The path(s) to the dataset to be converted to HATS.
    output_path
        The path where the output HATS dataset will be saved.
    schema
        The schema of the input dataset. Can be a Schema object, a path to a schema
        file, or a dictionary representing the schema. If the schema is not set,
        the function will look for the ``_convert.yaml`` file in the output
        directory and, if it exists, will read the schema from there.
    format
        The format of the output files. If not provided, it will be inferred from
        the files in ``data_path``.
    hats_initial_order
        The initial HEALPix order to use for partitioning the data. Defaults to 6.
    hats_max_order
        The maximum HEALPix order to use for partitioning the data. If not
        provided, the tiling process will continue until all pixels contain
        fewer than ``hats_max_rows_per_file`` rows.
    hats_max_rows_per_pixel
        The maximum number of rows per pixel. If set to ``None``, no adaptive
        tiling will be performed and the data will be partitioned using the
        ``hats_initial_order``.
    partition_max_rows_per_file
        The maximum number of rows per partitioned file.
    partition_approximate_bytes_per_file
        The approximate number of bytes per partitioned file. Defaults to 1 GB.
    apply_callback
        A callback function or list of callback functions to apply to the data
        before saving. Each callback should take a Polars LazyFrame as input
        and return a modified LazyFrame.
    low_memory_strategy
        The strategy to use to prevent OOM errors. Options are ``'data_page_size'``,
        which reduces the data page size used when writing Parquet files;
        and ``'streaming_chunk_size'``, which set a more conservative streaming chunk
        size. A list of strategies can be provided, as well as a dictionary mapping
        strategies to their appropriate parameters (e.g.
        ``{'data_page_size': 512 * 1024 * 1024, 'healpix_4': None}``).
    convert_kwargs
        Additional keyword arguments to pass to the conversion function.
    preserve_progress
        If ``True``, will try to skip steps that have already been completed.
        Otherwise, will run all the steps of the conversion process.
    show_progress
        Track conversion progress and show a progress bar. Progress tracking
        can be inaccurate. For the initial pixelation, in particular, the estimate
        will only be accurate if the input data is approximately all-sky.
    log
        An optional logger to use for logging the conversion process.
    quiet
        If ``True``, suppresses logging output. Defaults to ``False``.

    """

    # Get logger
    log = log or get_xmatch_rs_logger(log_level=40 if quiet else 20)

    # Make sure data_path is a list of paths.
    if isinstance(data_path, (str, pathlib.Path)):
        data_path = [str(data_path)]
    else:
        data_path = [str(p) for p in data_path]

    # Create output directory if it doesn't exist.
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    convert_kwargs = convert_kwargs or {}

    # Load the schema.
    if schema is not None and not isinstance(schema, Schema):
        schema = Schema(schema)

    # Parse low-memory strategies.
    if isinstance(low_memory_strategy, str):
        low_memory_strategy = [low_memory_strategy]
    elif low_memory_strategy is None:
        low_memory_strategy = []

    for strategy in low_memory_strategy:
        if strategy not in _LOW_MEMORY_DEFAULTS:
            raise ValueError(f"Invalid low_memory_strategy {strategy!r}.")

        if strategy == "data_page_size":
            page_size = get_low_memory_strategy_value(low_memory_strategy, strategy)
            log.info(
                "Using low-memory strategy 'data_page_size' "
                f"with page size {page_size}.",
            )

            if "sink_parquet_kwargs" in convert_kwargs:
                convert_kwargs["sink_parquet_kwargs"]["data_page_size"] = page_size
            else:
                convert_kwargs["sink_parquet_kwargs"] = {"data_page_size": page_size}

        elif strategy == "streaming_chunk_size":
            chunk_size = get_low_memory_strategy_value(low_memory_strategy, strategy)
            polars.Config.set_streaming_chunk_size(chunk_size)
            log.info(
                "Using low-memory strategy 'streaming_chunk_size' "
                f"with chunk size {chunk_size}."
            )

    # Try to load the _convert.yaml file or create a new instance.
    if (output_path / "_convert.yaml").exists():
        log.info(f"Loading _convert.yaml from {output_path!s}.")
        convert_dict = read_yaml_file(output_path / "_convert.yaml", return_class=dict)
        convert_model = ConvertToHATSModel.model_validate(convert_dict)
        if schema is not None:
            convert_model.dataset_schema = schema.schema
    else:
        if schema is None:
            raise ValueError(
                "No schema provided and no _convert.yaml file found in the output "
                "directory. Please provide a schema for the input data.",
            )

        # Build the conversion model and the initial phase and
        # write the _convert.yaml file to the output directory.

        healpix_min_order_model = PhaseModel(
            name=f"healpix_{hats_initial_order}_partitioning",
            partitioning_type="full",
            description=f"Pixelation to order {hats_initial_order}.",
            healpix_order_0=None,
            healpix_order_1=hats_initial_order,
        )

        convert_model = ConvertToHATSModel(
            dataset_schema=schema.schema,
            output_path=output_path,
            complete=False,
            phases=[healpix_min_order_model],
        )
        convert_model.write()

    for phase in convert_model.phases:
        _run_hats_phase(
            data_path=data_path,
            model=convert_model,
            phase_name=phase.name,
            format=format,
            partition_max_rows_per_file=partition_max_rows_per_file,
            partition_approximate_bytes_per_file=partition_approximate_bytes_per_file,
            apply_callback=apply_callback,
            convert_kwargs=convert_kwargs,
            log=log,
            show_progress=show_progress,
            preserve_progress=preserve_progress,
        )


def _run_hats_phase(
    data_path: DATA_PATH_T,
    model: ConvertToHATSModel,
    phase_name: str,
    format: Literal["csv", "parquet"] | None,
    partition_max_rows_per_file: int | None,
    partition_approximate_bytes_per_file: int | Literal["auto"] | None,
    apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None,
    convert_kwargs: dict | None,
    log: SDSSLogger,
    show_progress: bool,
    preserve_progress: bool,
) -> None:
    """Runs the first phase of the HATS conversion: convert to a fixed tiling order."""

    schema = model.get_schema()
    phase = model.get_phase(phase_name)

    if preserve_progress and phase.complete:
        log.info(f"Phase {phase_name!r} already completed. Skipping.")
        return

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

    order_0 = phase.healpix_order_0
    order_1 = phase.healpix_order_1

    assert order_1 is not None, "healpix_order_1 must be set for the initial phase."

    if order_0 is None:
        _run_min_order_phase(
            data_path=data_path,
            model=model,
            schema=schema,
            phase_name=phase_name,
            format=format,
            order_1=order_1,
            partition_max_rows_per_file=partition_max_rows_per_file,
            partition_approximate_bytes_per_file=partition_approximate_bytes_per_file,
            apply_callback=apply_callback,
            convert_kwargs=convert_kwargs,
            log=log,
            show_progress=show_progress,
        )


def _run_min_order_phase(
    data_path: DATA_PATH_T,
    model: ConvertToHATSModel,
    schema: Schema,
    phase_name: str,
    format: Literal["csv", "parquet"] | None,
    order_1: int,
    partition_max_rows_per_file: int | None,
    partition_approximate_bytes_per_file: int | Literal["auto"] | None,
    apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None,
    convert_kwargs: dict | None,
    log: SDSSLogger,
    show_progress: bool,
) -> None:

    # Get the file provider for HATS files.
    hats_file_provider = get_hats_file_provider(
        order_1,
        column_name=f"_healpix_{order_1}",
        use_uuid=True,
    )

    # The output path for the Norder hive files, followong the HATS convention.
    output_path = model.output_path / "dataset"

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
                partition_max_rows_per_file=partition_max_rows_per_file,
                partition_approximate_bytes_per_file=partition_approximate_bytes_per_file,
                partition_include_key=False,
                parallelise=False,
                healpix_orders=[29, order_1],
                healpix_column_name="_healpix_{order}",
                apply_callback=apply_callback,
                show_progress=show_progress,
                log=log,
                **(convert_kwargs or {}),
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
                partition_max_rows_per_file=partition_max_rows_per_file,
                partition_approximate_bytes_per_file=partition_approximate_bytes_per_file,
                partition_include_key=False,
                parallelise=True,
                healpix_orders=[29, order_1],
                healpix_column_name="_healpix_{order}",
                apply_callback=apply_callback,
                log=log,
                show_progress=show_progress,
                **(convert_kwargs or {}),
            )

        else:
            raise ValueError(
                f"Invalid format {format!r}. Currently supported formats "
                "are 'csv' and 'parquet'.",
            )

        log.info(f"Phase {phase_name!r} completed in {timer.elapsed:.1f} seconds.")

        model.update_phase(phase_name, complete=True)
