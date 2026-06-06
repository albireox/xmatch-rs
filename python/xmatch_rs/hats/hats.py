#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-04-07
# @Filename: hats.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import shutil

from typing import TYPE_CHECKING, Literal

import sdsstools.logger

import xmatch_rs._xmatch_rs_lib.hats as _rs_hats
from xmatch_rs.hats.margin import create_margin_cache
from xmatch_rs.hats.metadata import create_partition_info, create_properties
from xmatch_rs.hats.models import HATSPropertiesModel
from xmatch_rs.hats.object import convert_to_hats_object_catalogue
from xmatch_rs.schema import Schema
from xmatch_rs.tools import get_xmatch_rs_logger


if TYPE_CHECKING:
    from xmatch_rs.types import APPLY_CALLBACK_T


class HATS:
    """A representation of a dataset in the HATS format."""

    def __init__(self, path: str | pathlib.Path):
        """Initializes the HATS dataset.

        Parameters
        ----------
        path
            The path to a HATS collection.

        """

        self.path = pathlib.Path(path)

        if not self.path.exists():
            raise ValueError(f"The provided path {self.path!r} does not exist.")
        elif not self.path.is_dir():
            raise ValueError(f"The provided path {self.path!r} is not a directory.")

    def __repr__(self) -> str:
        return f"<HATS (path={str(self.path)!r})>"

    @property
    def schema(self) -> Schema:
        """Returns the schema of the HATS collection."""

        schema_path = self.path / "_schema.yaml"

        if not schema_path.exists():
            raise FileNotFoundError(f"No schema file found at {schema_path!s}.")

        return Schema(schema_path)

    @property
    def properties(self) -> HATSPropertiesModel:
        """Returns the properties of the HATS collection."""

        properties_path = self.path / "collection.properties"

        if not properties_path.exists():
            raise FileNotFoundError(f"No properties file found at {properties_path!s}.")

        return HATSPropertiesModel.read(properties_path)

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
        rechunk: bool = True,
        max_rows_per_file: int | None = None,
        max_kilobytes_per_file: int | None = None,
        create_primary_key_index: bool = True,
        primary_key_column: str | None = None,
        extra_index_columns: list[str] = [],
        check_index_duplicates: bool = False,
        sort_index: bool = False,
        n_workers: int | None = None,
        dynamic_planning: bool = True,
        apply_callback: APPLY_CALLBACK_T | list[APPLY_CALLBACK_T] | None = None,
        show_progress: bool = True,
        log: sdsstools.logger.SDSSLogger | None = None,
        quiet: bool = False,
        overwrite: bool = False,
    ) -> HATS:
        """Converts a dataset to the HATS object catalogue format.

        Parameters
        ----------
        data_path
            The path(s) to the input dataset. Can be a single path or a list of paths.
        collection_path
            The path where the output HATS collecton will be created. The resulting
            directory structure will be ``<collection_path>/object/dataset/...``
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
        rechunk
            If ``True``, the dataset will be rechunked to ensure that each file
            contains at most ``max_rows_per_file`` rows and is approximately smaller
            than ``max_kilobytes_per_file`` kilobytes.
        max_rows_per_file
            The maximum number of rows per file when rechunking. Only used if
            ``rechunk`` is ``True``. If not provided, the original file chunks
            created during the tiling process will be used, which ensures that
            each file contains at most ``max_rows_per_pixel`` rows.
        max_kilobytes_per_file
            The maximum number of kilobytes per file when rechunking. Only used if
            ``rechunk`` is ``True``. If not provided, the original file chunks
            created during the tiling process will be used, which ensures that
            each file is approximately smaller than ``max_kilobytes_per_pixel``
            kilobytes.
        create_primary_key_index
            If ``True``, a HATS index dataset will be created for the primary key
            column defined in the schema.
        primary_key_column
            The column to use as the primary key for the index. If not provided, the
            primary key column defined in the schema will be used.
        extra_index_columns
            A list of additional columns to include in the primary key index dataset.
        check_index_duplicates
            Whether to check for duplicate entries in the primary key index.
            This may lead to high memory use for large datasets.
        sort_index
            If ``True``, sorts the index by the index column. Sorting can cause OOM
            errors for large datasets, so it is disabled by default.
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
            If ``True``, suppresses logging output. Defaults to ``False``. Ignored
            if a logger is provided.
        overwrite
            If ``True``, allows overwriting existing files in the output directory.
            Defaults to ``False``.

        Returns
        -------
        HATS
            An instance of the HATS class representing the converted dataset.

        """

        log = log or get_xmatch_rs_logger(log_level=40 if quiet else 20)

        # Load the schema.
        if schema is not None and not isinstance(schema, Schema):
            schema = Schema(schema)

        convert_model = convert_to_hats_object_catalogue(
            data_path=data_path,
            schema=schema,
            collection_path=collection_path,
            format=format,
            initial_order=initial_order,
            reference_order=reference_order,
            max_order=max_order,
            max_rows_per_pixel=max_rows_per_pixel,
            max_kilobytes_per_pixel=max_kilobytes_per_pixel,
            rechunk=rechunk,
            max_rows_per_file=max_rows_per_file,
            max_kilobytes_per_file=max_kilobytes_per_file,
            n_workers=n_workers,
            dynamic_planning=dynamic_planning,
            apply_callback=apply_callback,
            show_progress=show_progress,
            log=log,
            overwrite=overwrite,
        )

        if schema is None:
            schema = convert_model.get_schema()

        # Write the schema file to the root of the collection.
        collection_path = pathlib.Path(collection_path)
        schema_path = collection_path / "_schema.yaml"

        if schema_path.exists() and not overwrite:
            raise FileExistsError(f"The schema file {schema_path!s} already exists.")

        schema.write_schema(schema_path)

        # Add the properties file for the object catalogue.
        create_properties(
            "object",
            collection_path,
            schema=schema,
            overwrite=overwrite,
        )
        log.info(f"Object catalogue properties file created at {collection_path!s}.")

        # Add the properties for the collection.
        create_properties(
            "collection",
            collection_path,
            schema=schema,
            overwrite=overwrite,
        )
        log.info(f"Collection properties file created at {collection_path!s}.")

        # Add the partition info file for the object catalogue.
        create_partition_info(
            "object",
            collection_path / "object",
            overwrite=overwrite,
        )
        log.info("Object catalogue partition info file created.")

        # Create the primary key index if requested.
        obj = cls(pathlib.Path(collection_path))

        if create_primary_key_index:
            if primary_key_column is not None:
                pk = primary_key_column
            else:
                pk = schema.get_primary_key()

            log.info(f"Creating primary key index for column {pk!r}.")
            obj.create_index(
                index_name=f"index_{pk}",
                index_column=pk,
                extra_columns=extra_index_columns,
                check_duplicates=check_index_duplicates,
                overwrite=overwrite,
                mode="full" if check_index_duplicates else "sequential",
                max_rows_per_file=max_rows_per_file,
                max_kilobytes_per_file=max_kilobytes_per_file,
                sort=sort_index,
            )

        return obj

    def rechunk(
        self,
        product: str,
        product_type: Literal["object", "margin", "index"],
        max_rows_per_file: int | None = None,
        max_kilobytes_per_file: int | None = None,
    ):
        """Rechunks a HATS product.

        Parameters
        ----------
        product
            The HATS product to rechunk (e.g., "object", "margin_5_arcsec").
        product_type
            The type of the product to rechunk. Must be one of "object", "margin", or
            "index". Used to determine whether the dataset is organised by HEALPix
            pixels.
        max_rows_per_file
            The maximum number of rows per file when rechunking. If not provided,
            defaults to 1,000,000 rows per file when ``max_kilobytes_per_file`` is
            provided.
        max_kilobytes_per_file
            The maximum number of kilobytes per file when rechunking.

        """

        if product_type in ["object", "margin"]:
            product_path = self.path / product
            is_dataset = True
        elif product_type == "index":
            product_path = self.path / product / "dataset" / "index"
            is_dataset = False

        if not product_path.exists():
            raise ValueError(f"The specified product {product!s} does not exist.")
        elif not product_path.is_dir():
            raise ValueError(f"The specified product {product!s} is not a directory.")

        if max_rows_per_file is None and max_kilobytes_per_file is None:
            raise ValueError(
                "At least one of max_rows_per_file or "
                "max_kilobytes_per_file must be provided."
            )

        _rs_hats.rechunk(
            product_path,
            is_dataset=is_dataset,
            max_rows_per_file=max_rows_per_file,
            approximate_bytes_per_file=max_kilobytes_per_file * 1024
            if max_kilobytes_per_file is not None
            else None,
        )

    def create_index(
        self,
        index_name: str,
        index_column: str,
        extra_columns: list[str] = [],
        check_duplicates: bool = False,
        sort: bool = False,
        max_rows_per_file: int | None = None,
        max_kilobytes_per_file: int | None = None,
        mode: Literal["sequential", "full"] = "sequential",
        overwrite: bool = False,
    ):
        """Adds an index to the HATS object catalogue.

        Parameters
        ----------
        index_name
            The name of the index to add. The index dataset will be created under the
            directory ``<collection_path>/<index_name>/dataset/index``.
        index_column
            The column to index on.
        extra_columns
            Any additional columns to include in the index.
        check_duplicates
            Whether to check for duplicate entries in the index. If ``True``, an error
            will be raised if duplicates are found. This can significantly slow down the
            index generation for large datasets.
        sort
            If ``True``, sorts the index by the index column. Sorting can cause OOM
            errors for large datasets, so it is disabled by default.
        max_rows_per_file
            The maximum number of rows per file when chunking the index.
            Defaults to 10 million rows per file.
        max_kilobytes_per_file
            The maximum number of kilobytes per file when chunking the index.
            Defaults to 10 GB.
        mode
            The mode to use for creating the index. If ``sequential``, the index
            will be created by reading the parquet files in each pixel sequentially.
            This method uses less memory but can be slower for large datasets.
            If ``full``, the entire dataset will be scanned and the index will be
            created using the streaming engine. ``check_duplicates`` can only be used
            in ``full`` mode.
        overwrite
            If ``True``, allows overwriting existing files in the output directory.
            Defaults to ``False``.

        """

        if mode not in ["sequential", "full"]:
            raise ValueError("Invalid mode. Must be one of 'sequential' or 'full'.")

        if check_duplicates and mode != "full":
            raise ValueError("Duplicate checking can only be performed in 'full' mode.")

        max_rows_per_file = max_rows_per_file or 10_000_000
        max_kilobytes_per_file = max_kilobytes_per_file or 10 * 1024 * 1024

        # Check paths.
        object_path = self.path / "object"
        if not object_path.exists():
            raise FileNotFoundError(f"No object catalogue found at {object_path!s}.")

        index_path = self.path / index_name / "dataset" / "index"
        if index_path.exists():
            if not overwrite:
                raise FileExistsError(f"The index {index_name!r} already exists.")
            else:
                shutil.rmtree(index_path)

        # Create the index dataset.
        _rs_hats.create_index(
            collection_path=self.path,
            index_name=index_name,
            index_column=index_column,
            extra_columns=extra_columns,
            mode=mode,
            check_duplicates=check_duplicates,
            sort=sort,
            max_rows_per_file=max_rows_per_file,
            max_kilobytes_per_file=max_kilobytes_per_file,
        )

        # Add the properties file for the index.
        create_properties(
            "index",
            self.path,
            data_path=self.path / index_name,
            schema=self.schema,
            overwrite=overwrite,
            data_fields={
                "hats_index_column": index_column,
                "hats_index_extra_column": " ".join(extra_columns),
            },
        )

        # Modify the colection properties to include the new index.
        # The all_indexes field is a space-separated map of the index column
        # and the index directory.
        coll_properties = self.properties
        if coll_properties.all_indexes is None:
            coll_properties.all_indexes = f"{index_column} {index_name}"
        else:
            coll_properties.all_indexes += f" {index_column} {index_name}"

        coll_properties.write(self.path / "collection.properties")

    def create_margin_cache(
        self,
        margin_name: str,
        buffer: float,
        overwrite: bool = False,
        n_workers: int | None = None,
        show_progress: bool = True,
        default_margin: bool = False,
        log: sdsstools.logger.SDSSLogger | None = None,
        quiet: bool = False,
    ):
        """Creates a margin cache for the HATS object catalogue.

        A margin cache is a dataset that contains additional sources around the edges
        of each HATS partition pixel. This is useful for cross-matching with other
        datasets, as it allows sources that are close to the edge of a pixel to be
        included in the cross-match.

        The margin cache will be created under the directory
        ``<collection_path>/<margin_name>/dataset``. Each file in the margin cache will
        contain sources from the object catalogue that are within a certain distance
        (the buffer) from the edge of the corresponding HATS partition pixel.

        Somewhat counterintuitively, smaller buffer values can increase the
        margin cache creation time. This is because the total runtime is dominated
        by the time required to create the BMOC for each pixel. A smaller buffer
        requires a higher BMOC order, which makes BMOC creation slower.

        Parameters
        ----------
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
        default_margin
            If ``True``, sets this margin cache as the default margin for the
            collection. If this is the first margin cache being created for the
            collection, it will be set as the default margin regardless of the value
            of this parameter.
        log
            An optional logger to use for logging the margin cache creation process.
        quiet
            If ``True``, suppresses logging output. Defaults to ``False``. Ignored
            if a logger is provided.

        """

        log = log or get_xmatch_rs_logger(log_level=40 if quiet else 20)

        collection_path = self.path
        margin_path = collection_path / margin_name

        create_margin_cache(
            collection_path=collection_path,
            margin_name=margin_name,
            buffer=buffer,
            overwrite=overwrite,
            n_workers=n_workers,
            show_progress=show_progress,
            log=log,
        )

        # Add the partition info file for the margin cache.
        create_partition_info(
            "margin",
            margin_path,
            overwrite=overwrite,
        )
        log.info("Margin cache partition info file created.")

        # Add the properties file for the margin cache.
        create_properties(
            "margin",
            collection_path,
            data_path=margin_path,
            schema=self.schema,
            overwrite=overwrite,
            data_fields={"hats_margin_threshold": str(buffer)},
        )
        log.info("Margin properties file created.")

        # Update the collection properties to include the new margin cache.
        coll_properties = self.properties
        if coll_properties.all_margins is None:
            coll_properties.all_margins = str(margin_path.name)
        else:
            coll_properties.all_margins += f" {margin_path.name}"

        if coll_properties.default_margin is None or default_margin:
            coll_properties.default_margin = margin_path.name

        coll_properties.write(self.path / "collection.properties")

        log.info("Updated margin cache in collection properties.")
