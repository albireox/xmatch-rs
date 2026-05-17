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
    from xmatch_rs.types import APPLY_CALLBACK_T


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
        collection_path: str | pathlib.Path,
        schema: Schema | str | pathlib.Path | dict | None = None,
        format: Literal["csv", "parquet"] | None = None,
        initial_order: int = 2,
        reference_order: int = 29,
        max_order: int | None = None,
        max_rows_per_pixel: int | None = 1_000_000,
        max_kilobytes_per_pixel: int | None = None,
        n_workers: int | None = None,
        dynamic_planning: bool = True,
        apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None = None,
        show_progress: bool = True,
        log: sdsstools.logger.SDSSLogger | None = None,
        quiet: bool = False,
    ) -> HATS:
        """Converts a dataset to the HATS format.

        Parameters
        ----------
        data_path
            The path(s) to the input dataset. Can be a single path or a list of paths.
        collection_path
            The path where the output HATS collecton will be created. The resulting
            directory structure will be ``<collection_path>/catalog/dataset/...``
            along with the necessary metadata files.
        schema
            The schema of the input dataset. Can be a Schema object, a path to a schema
            file, or a dictionary representing the schema. If the schema is not set,
            the function will look for the ``_convert.yaml`` file in the output
            directory and, if it exists, will read the schema from there.
        format
            The format of the output files. If not provided, it will be inferred from
            the files in ``path``.
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
            ``max_kilobytes_per_pixel`` are not exceeded. If ``False``,
            a conversion YAML file is expected to be present in the output directory
            and the conversion process will follow the phases defined in that file.
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
            If ``True``, suppresses logging output. Defaults to ``False``.

        Returns
        -------
        HATS
            An instance of the HATS class representing the converted dataset.

        """

        convert_to_hats(
            data_path=data_path,
            schema=schema,
            collection_path=collection_path,
            format=format,
            initial_order=initial_order,
            reference_order=reference_order,
            max_order=max_order,
            max_rows_per_pixel=max_rows_per_pixel,
            max_kilobytes_per_pixel=max_kilobytes_per_pixel,
            n_workers=n_workers,
            dynamic_planning=dynamic_planning,
            apply_callback=apply_callback,
            show_progress=show_progress,
            log=log,
            quiet=quiet,
        )

        return cls(pathlib.Path(collection_path))
