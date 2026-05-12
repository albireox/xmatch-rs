#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-04-07
# @Filename: hats.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import TYPE_CHECKING, Literal

import sdsstools.logger

from xmatch_rs.convert.hats import convert_to_hats
from xmatch_rs.schema import Schema


if TYPE_CHECKING:
    from xmatch_rs.types import APPLY_CALLBACK_T, LOW_MEMORY_STRATEGY_T


class HATS:
    """A representation of a dataset in the HATS format."""

    def __init__(self, path: str | pathlib.Path):
        """Initializes the HATS dataset.

        Parameters
        ----------
        path
            The path to a dataset structured such that it follows the
            HATS specification.

        """

        self.path = pathlib.Path(path)

        if not self.path.exists():
            raise ValueError(f"The provided path {self.path!r} does not exist.")
        elif not self.path.is_dir():
            raise ValueError(f"The provided path {self.path!r} is not a directory.")

    def __repr__(self) -> str:
        return f"<HATS (path={str(self.path)!r})>"

    @classmethod
    def convert_to_hats(
        cls,
        data_path: list[str | pathlib.Path] | str | pathlib.Path,
        output_path: str | pathlib.Path,
        schema: Schema | str | pathlib.Path | dict | None = None,
        format: Literal["csv", "parquet"] | None = None,
        hats_initial_order: int = 2,
        hats_max_order: int | None = None,
        hats_max_rows_per_pixel: int | None = None,
        partition_max_rows_per_file: int | None = None,
        partition_approximate_bytes_per_file: int | Literal["auto"] | None = None,
        apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None = None,
        low_memory_strategy: LOW_MEMORY_STRATEGY_T | None = None,
        convert_kwargs: dict | None = None,
        preserve_progress: bool = True,
        show_progress: bool = True,
        log: sdsstools.logger.SDSSLogger | None = None,
        quiet: bool = False,
    ) -> HATS:
        """Converts a dataset to the HATS format.

        Parameters
        ----------
        data_path
            The path(s) to the input dataset. Can be a single path or a list of paths.
        output_path
            The path where the output HATS dataset will be saved.
        schema
            The schema of the input dataset. Can be a Schema object, a path to a schema
            file, or a dictionary representing the schema. If the schema is not set,
            the function will look for the ``_convert.yaml`` file in the output
            directory and, if it exists, will read the schema from there.
        format
            The format of the output files. If not provided, it will be inferred from
            the files in ``path``.
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
            and ``'streaming_chunk_size'``, which set a more conservative streaming
            chunk size. A list of strategies can be provided, as well as a dictionary
            mapping strategies to their appropriate parameters (e.g.
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

        Returns
        -------
        HATS
            An instance of the HATS class representing the converted dataset.

        """

        convert_to_hats(
            data_path=data_path,
            schema=schema,
            output_path=output_path,
            format=format,
            hats_initial_order=hats_initial_order,
            hats_max_order=hats_max_order,
            hats_max_rows_per_pixel=hats_max_rows_per_pixel,
            partition_max_rows_per_file=partition_max_rows_per_file,
            partition_approximate_bytes_per_file=partition_approximate_bytes_per_file,
            apply_callback=apply_callback,
            convert_kwargs=convert_kwargs,
            preserve_progress=preserve_progress,
            low_memory_strategy=low_memory_strategy,
            show_progress=show_progress,
            log=log,
            quiet=quiet,
        )

        return cls(pathlib.Path(output_path))
