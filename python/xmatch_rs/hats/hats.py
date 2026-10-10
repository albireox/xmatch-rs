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

from typing import TYPE_CHECKING, Iterator, Literal, Sequence

import numpy
import polars
from healpy.newvisufunc import projview
from matplotlib import pyplot as plt
from rich.live import Live

import sdsstools.logger

import xmatch_rs._xmatch_rs_lib.hats as _rs_hats
from xmatch_rs.hats.margin import create_margin_cache
from xmatch_rs.hats.metadata import create_partition_info, create_properties
from xmatch_rs.hats.models import HATSPropertiesModel
from xmatch_rs.hats.object import convert_to_hats_object_catalogue
from xmatch_rs.hats.tools import calculate_max_proper_motion
from xmatch_rs.schema import Schema
from xmatch_rs.tools import get_xmatch_rs_logger


if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from xmatch_rs.types import APPLY_CALLBACK_T


__all__ = ["HATS", "HATSProduct", "HATSStore"]


class HATS:
    """A representation of a HATS collection."""

    def __init__(self, path: str | pathlib.Path):
        """A representation of a HATS collection.

        Parameters
        ----------
        path
            The path to a HATS collection.

        """

        self.path = pathlib.Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"The path {self.path!r} does not exist.")
        elif not self.path.is_dir():
            raise FileNotFoundError(f"The path {self.path!r} is not a directory.")

    def __repr__(self) -> str:
        return f"HATS(path={self.path!s})"

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

    def validate(self, check_rows: bool = False):
        """Validates the HATS collection.

        Parameters
        ----------
        check_rows
            If True, also checks the number of rows within the products
            for consistency.

        """

        if not (self.path / "object").exists():
            raise ValueError("The HATS collection must contain an 'object' product.")

        if not (self.path / "collection.properties").exists():
            raise ValueError("Missing 'collection.properties' file.")

        if not (self.path / "_schema.yaml").exists():
            raise ValueError("Missing '_schema.yaml' file.")

        if check_rows:
            self.get_product("object")._validate_nrows()

        properties = self.properties
        if properties.all_margins is not None:
            for margin in properties.all_margins.split(" "):
                if not (self.path / margin).exists():
                    raise ValueError(
                        f"Margin {margin!r} listed in collection properties but "
                        f"not found in collection."
                    )

                if check_rows:
                    self.get_product(margin)._validate_nrows()

        if properties.all_indexes is not None:
            for index in properties.all_indexes.split(" ")[1::2]:
                if not (self.path / index).exists():
                    raise ValueError(
                        f"Index {index!r} listed in collection properties but "
                        f"not found in collection."
                    )

                if check_rows:
                    self.get_product(index)._validate_nrows()

    def get_product(self, product: str) -> HATSProduct:
        """Returns the specified product from the HATS collection."""

        product_path = self.path / product
        if not product_path.exists():
            raise FileNotFoundError(f"Product {product!r} not found in the collection.")

        indices = (
            self.properties.all_indexes.split(" ")[1::2]
            if self.properties.all_indexes is not None
            else []
        )
        margins = (
            self.properties.all_margins.split(" ")
            if self.properties.all_margins is not None
            else []
        )

        if product == "object":
            return HATSProduct(self, "object", "object")
        elif product in indices:
            return HATSProduct(self, "index", product)
        elif product in margins:
            return HATSProduct(self, "margin", product)

        raise ValueError(f"Invalid product type: {product!r}")

    def list_products(self) -> list[str]:
        """Lists the products available in the HATS collection."""

        products = []

        # If the collection validates it must have an object product.
        self.validate()
        products.append("object")

        properties = self.properties
        if properties.all_margins is not None:
            for margin in properties.all_margins.split(" "):
                products.append(margin)

        if properties.all_indexes is not None:
            for index in properties.all_indexes.split(" ")[1::2]:
                products.append(index)

        return products

    def get_as_lazyframe(self, product: str = "object") -> polars.LazyFrame:
        """Returns a lazy frame for a given product."""

        return self.get_product(product).as_lazyframe()

    @classmethod
    def convert_to_hats(
        cls,
        data_path: list[str | pathlib.Path] | str | pathlib.Path,
        collection_path: str | pathlib.Path,
        schema: Schema | str | pathlib.Path | dict | None = None,
        format: Literal["csv", "parquet", "fits"] | None = None,
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
        create_density_map: bool = True,
        density_map_order: int = 11,
        density_map_kwargs: dict | None = None,
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
            the function will look for the ``_schema.yaml`` file in the collection
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
        create_density_map
            If ``True``, a density map will be created for the dataset.
        density_map_order
            The order of the density map.
        density_map_kwargs
            Additional keyword arguments to pass to the density map creation function.
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
        elif schema is None:
            schema_path = pathlib.Path(collection_path) / "_schema.yaml"
            if not schema_path.exists():
                raise ValueError(
                    "No schema provided and no '_schema.yaml' file found in "
                    f"{collection_path!s}."
                )
            schema = Schema(schema_path)

        convert_to_hats_object_catalogue(
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
            collection_path / "object",
            overwrite=overwrite,
            write_csv=True,
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

        if create_density_map:
            log.info("Creating density map.")
            obj.create_density_map(
                order=density_map_order,
                coord="galactic",
                **(density_map_kwargs or {}),
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
        buffer: float | None = None,
        overwrite: bool = False,
        n_workers: int | None = None,
        show_progress: bool = True,
        default_margin: bool = False,
        epoch: float | None = None,
        pm_percentile: float = 100.0,
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
            sources will be included in the margin cache. If not provided, calculates
            the buffer based on the maximum proper motion for the given ``epoch``.
            It is possible to provide a specific ``pm_percentile`` to reject the
            highest proper motion objects.
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
        epoch
            The epoch, in Julian years, for which to calculate the ``buffer``. Ignored
            if ``buffer`` is provided.
        pm_percentile
            The percentile of the proper motion distribution to use for calculating the
            ``buffer``. Defaults to 100.0, which means the maximum proper motion is
            used.
        log
            An optional logger to use for logging the margin cache creation process.
        quiet
            If ``True``, suppresses logging output. Defaults to ``False``. Ignored
            if a logger is provided.

        """

        log = log or get_xmatch_rs_logger(log_level=40 if quiet else 20)

        collection_path = self.path
        margin_path = collection_path / margin_name

        if buffer is None:
            buffer = calculate_max_proper_motion(
                self,
                epoch=epoch,
                percentile=pm_percentile,
            )

            if buffer is None:
                raise RuntimeError(
                    "Cannot calculate the buffer for the margin cache. "
                    "This usually means that the catalogue does not provide proper "
                    "motions or that the catalogue schema is incomplete."
                )

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
            margin_path,
            overwrite=overwrite,
            write_csv=True,
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

    def create_density_map(
        self,
        map_order: int,
        coord: Literal["equatorial", "galactic"] = "equatorial",
        norm: str | None = None,
        m: Sequence[int] | None = None,
        output_path: str | pathlib.Path | None = None,
        **map_kwargs,
    ) -> ArrayLike:
        """Creates a density map for the HATS dataset.

        Parameters
        ----------
        map_order
            The order of the HEALPix map to generate.
        coord
            The coordinate system to use for the density map. Can be ``'equatorial'``
            or ``'galactic'``.
        norm
            The normalization to use for the density map.
        m
            The map to use for the density map (must be in the RING schema).
            If not provided it will be calculated from the HATS dataset.
        output_path
            The path to save the generated density map. If not provided, the map
            will be saved to the default location in the HATS object directory.
        map_kwargs
            Additional keyword arguments to pass to the ``healpy.newvisufunc.projview``
            function.

        Returns
        -------

        """

        if m is None:
            healpix_map = numpy.array(
                _rs_hats.get_healpix_map(
                    str(self.path / "object"),
                    map_order,
                )
            )
        else:
            healpix_map = numpy.array(m)

        if output_path is None:
            output_path = self.path / "object" / f"density_map_order_{map_order}.pdf"
        else:
            output_path = pathlib.Path(output_path)

        projview_kwargs = {
            "coord": "C",
            "cb_orientation": "vertical",
            "graticule": True,
            "graticule_labels": True,
            "projection_type": "mollweide",
            "unit": "Target count",
            "xlabel": "RA",
            "ylabel": "Dec",
            "norm": norm,
            "title": self.properties.obs_collection,
        }

        if coord == "galactic":
            projview_kwargs["coord"] = ["C", "G"]
            projview_kwargs["xlabel"] = "Galactic Longitude"
            projview_kwargs["ylabel"] = "Galactic Latitude"
        elif coord == "equatorial":
            pass
        else:
            raise ValueError(f"Unsupported coordinate system: {coord!r}")

        projview_kwargs.update(map_kwargs)

        with plt.ioff():
            quad = projview(m=healpix_map, **projview_kwargs)
            quad.figure.set_size_inches((16, 9))
            quad.figure.savefig(output_path)
            plt.close(quad.figure)

        return healpix_map

    def write_to_database(
        self,
        database_uri: str,
        show_progress: bool = True,
        n_workers: int | None = 4,
        **kwargs,
    ) -> None:
        """Loads the HATS dataset into a database.

        Parameters
        ----------
        database_uri
            The URI of the database to load into.
        show_progress
            Whether to show a progress bar while loading the data.
        n_workers
            The number of workers to use for loading the data. If None, uses all
            available CPU cores.
        kwargs
            Additional keyword arguments to pass to the
            :obj:`~xmatch_rs.database.load.write_dataset_to_database` function.

        """

        from xmatch_rs.database.load import write_dataset_to_database

        write_dataset_to_database(
            self,
            database_uri=database_uri,
            show_progress=show_progress,
            n_workers=n_workers,
            **kwargs,
        )

    def read_partition_info(self, product: str = "object") -> polars.DataFrame:
        """Reads the partition info file for a given product."""

        partition_info_path = self.path / product / "partition_info.csv"

        if not partition_info_path.exists():
            raise FileNotFoundError(
                f"No partition info file found for product {product!r}."
            )

        return polars.read_csv(
            partition_info_path,
            schema={"Norder": polars.Int8, "Npix": polars.Int64},
        )


class HATSProduct:
    """Base class for HATS products.

    Parameters
    ----------
    hats_collection
        The HATS collection to which this product belongs.
    product_type
        The type of the product.
    product_path
        The path to the product.

    """

    def __init__(
        self,
        hats_collection: HATS,
        product_type: Literal["object", "margin", "index"],
        product_name: str,
    ):
        self.product_name = product_name
        self.collection_path = hats_collection.path

        self.product_type = product_type
        if self.product_type not in ["object", "margin", "index"]:
            raise ValueError(f"Invalid product type: {self.product_type!r}")

        self.product_path = self.collection_path / product_name
        if not self.product_path.exists():
            raise FileNotFoundError(f"Product path not found: {self.product_path!s}")

        self.dataset_path = self.product_path / "dataset"
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path!s}")

    def __repr__(self) -> str:
        return (
            f"HATSProduct(product_type={self.product_type!r}, "
            f"product_name={self.product_name!r})"
        )

    @property
    def properties(self) -> HATSPropertiesModel:
        """Returns the properties model."""

        assert self.product_path is not None, "Product path must be specified."
        properties_file = self.product_path / "properties"
        if not properties_file.exists():
            raise FileNotFoundError(f"Properties file not found: {properties_file!s}")

        return HATSPropertiesModel.read(properties_file)

    def as_lazyframe(self) -> polars.LazyFrame:
        """Return a Polars LazyFrame with the product data."""

        return polars.scan_parquet(self.dataset_path)

    def _validate_nrows(self) -> None:
        """Checks that the number of rows in the dataset matches the properties file."""

        props_nrows = self.properties.hats_nrows
        if props_nrows is None:
            raise ValueError(
                f"Number of rows for product {self.product_name!r} is "
                "missing from the properties file."
            )

        dataset_lf = self.as_lazyframe()
        dataset_nrows: int = (
            dataset_lf.select(polars.len()).collect(engine="streaming").item()
        )

        if dataset_nrows != props_nrows:
            raise ValueError(
                f"Number of rows for product {self.product_name!r} does not match the "
                "value specified in the properties file."
            )


class HATSStore:
    """Represents a store of HATS collections, one for each catalogue."""

    def __init__(self, store_path: str | pathlib.Path):
        self.store_path = pathlib.Path(store_path)
        if not self.store_path.exists():
            raise FileNotFoundError(f"Store path not found: {self.store_path!s}")

    def __repr__(self) -> str:
        return f"HATSStore(store_path={self.store_path!s})"

    def list_collections(self) -> list[str]:
        """Lists all HATS collections available in the store."""

        return [
            pp.name
            for pp in self.store_path.iterdir()
            if pp.is_dir()
            and (pp / "_schema.yaml").exists()
            and (pp / "collection.properties").exists()
        ]

    def get_collection(self, collection_name: str) -> HATS:
        """Returns the specified HATS collection from the store."""

        if collection_name not in self.list_collections():
            raise ValueError(f"Collection {collection_name!r} not found in the store.")

        collection_path = self.store_path / collection_name
        if not collection_path.exists():
            raise FileNotFoundError(f"Collection {collection_name!r} not found.")

        return HATS(collection_path)

    def iter_collections(self) -> Iterator[HATS]:
        """Iterates over all HATS collections in the store."""

        for collection_name in self.list_collections():
            yield self.get_collection(collection_name)

    def validate_all(self, check_rows: bool = False) -> None:
        """Validates all HATS collections in the store."""

        for collection in self.iter_collections():
            # message = Text("[*] ", style="blue")
            # message.append(f"Validating collection {collection.path!r}.")

            with Live(
                f"[blue][*][/] Validating collection {collection.path!s}",
                transient=False,
                auto_refresh=True,
                refresh_per_second=10,
            ) as live:
                try:
                    collection.validate(check_rows=check_rows)
                    live.update(
                        f"[green][✔][/] Collection {collection.path!s}: "
                        "successfully validated."
                    )
                except Exception as e:
                    live.update(
                        f"[red][✖][/] Collection {collection.path!s}: "
                        f"validation failed: {e}"
                    )
