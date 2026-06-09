#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-05-17
# @Filename: metadata.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import Literal

import polars

import xmatch_rs
from xmatch_rs.hats.models import HATSPropertiesModel
from xmatch_rs.hats.object import get_dataset_df, get_dataset_nrows
from xmatch_rs.hats.tools import get_directory_size
from xmatch_rs.schema import Schema
from xmatch_rs.tools import get_current_time, lazy_frame_row_count


__all__: list[str] = ["HATSProperties", "create_properties", "create_partition_info"]


class HATSProperties:
    """A class to represent the properties metadata file for a HATS product."""

    def __init__(
        self,
        path: str | pathlib.Path,
        obs_collection: str | None = None,
        dataproduct_type: Literal[
            "object",
            "margin",
            "association",
            "index",
            "collection",
        ]
        | None = None,
        **fields,
    ):
        """Initializes the HATS properties.

        Parameters
        ----------
        path
            The path to the HATS properties file.
        obs_collection
            The short name of the original dataset (e.g., 2MASS). Required if the
            properties file does not exist.
        dataproduct_type
            The type of HATS product (e.g., "object", "margin", "association",
            "index", "collection"). Required if the properties file does not exist.
        fields
            Any additional fields to add to the properties file.

        """

        self.path = pathlib.Path(path)

        if self.path.name not in [
            "properties",
            "collection.properties",
            "hats.properties",
        ]:
            raise ValueError("Invalid file name for a properties file.")

        if not self.path.exists():
            if obs_collection is None or dataproduct_type is None:
                raise ValueError(
                    "The properties file does not exist. obs_collection and "
                    "dataproduct_type are required to create it."
                )
            self.model = HATSPropertiesModel(
                obs_collection=obs_collection,
                dataproduct_type=dataproduct_type,
                **fields,
            )
        else:
            try:
                model = HATSPropertiesModel.read(self.path)
            except Exception as err:
                raise ValueError("Failed to read properties file.") from err

            model_dump = model.model_dump()

            if obs_collection is not None:
                model_dump["obs_collection"] = obs_collection

            if dataproduct_type is not None:
                model_dump["dataproduct_type"] = dataproduct_type

            for key, value in fields.items():
                model_dump[key] = value

            self.model = HATSPropertiesModel(**model_dump)

    def write(self, overwrite: bool = False):
        """Writes the properties file to disk."""

        if self.path.exists() and not overwrite:
            raise ValueError(f"The properties file {self.path!r} already exists.")

        self.model.write(self.path)


def create_properties(
    dataproduct_type: Literal[
        "object",
        "margin",
        "association",
        "index",
        "collection",
    ],
    collection_path: str | pathlib.Path,
    data_path: str | pathlib.Path | None = None,
    schema: Schema | None = None,
    overwrite: bool = False,
    data_fields: dict[str, str] | None = None,
):
    """Creates a properties file for the corresponding HATS data product."""

    data_fields = data_fields or {}

    collection_path = pathlib.Path(collection_path)
    if not collection_path.exists():
        raise ValueError(f"The collection path {collection_path!r} does not exist.")

    if schema is None:
        schema_path = collection_path / "_schema.yaml"
        if not schema_path.exists():
            raise ValueError(f"The schema path {schema_path!s} does not exist.")
        schema = Schema(schema_path)

    # If we passed a data path, use it. Otherwise it will be set below.
    path = pathlib.Path(data_path) if data_path is not None else None

    if dataproduct_type == "object":
        object_path = collection_path / "object"
        if not object_path.exists():
            raise ValueError(f"The object path {object_path!s} does not exist.")

        path = path or (object_path / "properties")
    elif dataproduct_type == "collection":
        path = path or (collection_path / "collection.properties")
    elif dataproduct_type == "index":
        if data_path is None:
            raise ValueError("data_path is required for index properties.")
        path = pathlib.Path(data_path) / "properties"
    elif dataproduct_type == "margin":
        if data_path is None:
            raise ValueError("data_path is required for margin properties.")
        path = pathlib.Path(data_path) / "properties"
    else:
        raise NotImplementedError(f"Data type {dataproduct_type!r} is not supported.")

    if path.exists() and not overwrite:
        raise ValueError(f"The properties file {path!s} already exists.")

    obs_collection = schema.model.catalogue_name

    hats_builder = f"xmatch-rs v{xmatch_rs.__version__}"
    hats_creation_date = get_current_time(format="isot", timezone="UTC")
    hats_version = "1.0"

    hats_primary_table_url = schema.model.url
    bib_reference = schema.model.bib_reference
    bib_reference_url = schema.model.bib_reference_url

    hats_col_ra = None
    hats_col_dec = None
    hats_frame = None
    hats_coordinate_epoch = None
    hats_nrows = None
    hats_order = None
    hats_max_rows = None
    hats_estsize = None
    if dataproduct_type in ["object", "margin"]:
        hats_col_ra, hats_col_dec = schema.get_ra_dec_columns()
        hats_coordinate_epoch = schema.model.epoch
        hats_frame = "equatorial"

        hats_nrows = get_dataset_nrows(path.parent / "dataset")

        dataset_df = get_dataset_df(path.parent / "dataset", include_size=True)
        hats_order = dataset_df["Norder"].max()
        hats_max_rows = dataset_df["n_rows"].max()
        hats_estsize = dataset_df["size_kb"].sum()

    hats_index_column = None
    hats_index_extra_column = None
    if dataproduct_type == "index":
        hats_index_column = data_fields.pop("hats_index_column")
        if hats_index_column is None:
            raise ValueError(
                "hats_index_column is required for index properties. "
                "Please provide it in data_fields."
            )
        if "hats_index_extra_column" in data_fields:
            hats_index_extra_column = data_fields.pop("hats_index_extra_column")
            hats_index_extra_column = hats_index_extra_column.strip()

        dataset_path = path.parent / "dataset" / "index"
        hats_nrows = lazy_frame_row_count(polars.scan_parquet(dataset_path))
        hats_estsize = get_directory_size(dataset_path)

    properties = HATSProperties(
        path=path,
        header=f"HATS {dataproduct_type} properties",
        obs_collection=obs_collection,
        dataproduct_type=dataproduct_type,
        hats_primary_table_url=hats_primary_table_url,
        bib_reference=bib_reference,
        bib_reference_url=bib_reference_url,
        hats_col_ra=hats_col_ra,
        hats_col_dec=hats_col_dec,
        hats_coordinate_epoch=hats_coordinate_epoch,
        hats_frame=hats_frame,
        hats_nrows=hats_nrows,
        hats_order=str(hats_order) if hats_order is not None else None,
        hats_max_rows=str(hats_max_rows) if hats_max_rows is not None else None,
        hats_estsize=str(int(hats_estsize)) if hats_estsize is not None else None,
        hats_index_column=hats_index_column,
        hats_index_extra_column=hats_index_extra_column,
        hats_creation_date=hats_creation_date,
        hats_builder=hats_builder,
        hats_version=hats_version,
        **data_fields,
    )
    properties.write(overwrite=overwrite)


def create_partition_info(
    dataset_path: str | pathlib.Path,
    overwrite: bool = False,
    write_csv: bool = False,
) -> polars.DataFrame:
    """Creates the partition info file for the given product.

    Parameters
    ----------
    dataset_path
        The path to the HATS dataset.
    overwrite
        Whether to overwrite the partition info file if it already exists.
    write_csv
        Whether to write the partition info file as a CSV file.

    Returns
    -------
    DataFrame
        A DataFrame with the partition info.

    """

    dataset_path = pathlib.Path(dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"The dataset path {dataset_path!r} does not exist.")

    partition_info_path = dataset_path / "partition_info.csv"
    if partition_info_path.exists() and not overwrite and write_csv:
        raise ValueError(f"The file {partition_info_path!r} already exists.")

    data = get_dataset_df(dataset_path, include_size=False)
    data = data.select(["Norder", "Npix"]).sort(["Norder", "Npix"])

    if write_csv:
        data.write_csv(partition_info_path)

    return data
