#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-03-17
# @Filename: schema.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import warnings

from typing import Annotated

import polars
import yaml
from pydantic import BaseModel, Field

from sdsstools import read_yaml_file

import xmatch_rs
from xmatch_rs.tools import get_polars_dtype_from_string


__all__: list[str] = ["Schema", "XMATCH_RS_ROOT"]


XMATCH_RS_ROOT = pathlib.Path(xmatch_rs.__file__).parent


class SchemaModel(BaseModel):
    """Model representing a schema file."""

    catalogue_name: Annotated[
        str,
        Field(description="Short version of the name of the catalogue."),
    ]

    source_catalogue_url: Annotated[
        str,
        Field(description="A URL to the original source of the data."),
    ]

    bib_reference: Annotated[
        str | None,
        Field(description="A bibliographic reference for the catalogue."),
    ] = None

    bib_reference_url: Annotated[
        str | None,
        Field(description="A URL to a bibliographic reference for the catalogue."),
    ] = None

    comments: Annotated[
        str | None,
        Field(description="Any comments about the catalogue."),
    ] = None

    source_catalogue_params: Annotated[
        SourceCatalogueParamsModel | None,
        Field(description="Parameters to read the original source data."),
    ] = None

    ra_column: Annotated[
        str | None,
        Field(description="The name of the RA column in the catalogue (in degrees)."),
    ] = None

    dec_column: Annotated[
        str | None,
        Field(description="The name of the Dec column in the catalogue."),
    ] = None

    pmra_column: Annotated[
        str | None,
        Field(description="The name of the proper motion in RA column."),
    ] = None

    pmra_is_cosdec: Annotated[
        bool,
        Field(
            description="Whether the proper motion in RA column has already been "
            "deprojected by multiplying by cos(dec)."
        ),
    ] = True

    pmdec_column: Annotated[
        str | None,
        Field(description="The name of the proper motion in Dec column."),
    ] = None

    coordinate_epoch: Annotated[
        float | None,
        Field(description="The epoch of the coordinates as Julian year."),
    ] = None

    columns: Annotated[
        dict[str, ColumnModel],
        Field(
            description="The schema as a dictionary of "
            "column names and their data types."
        ),
    ]

    def get_source_columns(self) -> SchemaModel:
        """Returns an instance of the schema with only source columns."""

        clone = self.model_copy(deep=True)
        clone.columns = {
            column_name: column_data
            for column_name, column_data in clone.columns.items()
            if column_data.is_source_column
        }

        return clone


class SourceCatalogueParamsModel(BaseModel):
    """Model representing the parameters to read the original source data."""

    format: Annotated[
        str,
        Field(
            description="The format of the original source data. This should be a "
            "format that can be read by Polars, such as 'csv' or 'parquet'."
        ),
    ]

    csv_delimiter: Annotated[
        str | None,
        Field(
            description="The delimiter used in the original source data if it is in "
            "CSV format. This is optional but recommended if the format is CSV."
        ),
    ] = None

    csv_has_header: Annotated[
        bool | None,
        Field(
            description="Whether the original source data has a header row if it is "
            "in CSV format. This is optional but recommended if the format is CSV."
        ),
    ] = None

    csv_comment_prefix: Annotated[
        str | None,
        Field(
            description="The prefix used for comment lines in the original source "
            "data if it is in CSV format. This is optional but recommended if the "
            "format is CSV."
        ),
    ] = None


class SourceCatalogueParamsColumnModel(BaseModel):
    """Model representing the parameters of a column in the original source data."""

    column_name: Annotated[
        str | None,
        Field(description="The name of the column in the original source data."),
    ] = None

    dtype: Annotated[
        str | None,
        Field(
            description="The data type of the column in the original source data. "
            "Must be a string that can be converted to a Polars data type."
        ),
    ] = None

    cast_to: Annotated[
        str | None,
        Field(
            description="If the column in the original source data needs to be cast "
            "to a different data type, this field specifies the target data type. "
            "Must be a string that can be converted to a Polars data type."
        ),
    ] = None

    apply_functions: Annotated[
        list[str] | None,
        Field(
            description="A list of functions to apply to the column after reading it "
            "from the original source data. The functions are applied in the order "
            "they are listed. See the xmatch_rs.convert.callbacks module for available "
            "functions."
        ),
    ] = None


class DatabaseParamsModel(BaseModel):
    """Model representing database constraints and other parameters for a column."""

    unique: Annotated[
        bool,
        Field(description="Whether the values in the column are unique."),
    ] = False

    numerical_unique: Annotated[
        bool,
        Field(description="Whether the values in the column are unique and numerical."),
    ] = False

    primary_key: Annotated[
        bool,
        Field(description="Whether this column is the primary key for the catalogue."),
    ] = False


class ColumnModel(BaseModel):
    """Model representing a column in the schema."""

    dtype: Annotated[
        str,
        Field(
            description="The data type of the column. This should be a string "
            "that can be converted to a Polars data type."
        ),
    ]

    unit: Annotated[str | None, Field(description="The unit of the column.")] = None

    description: Annotated[
        str | None,
        Field(
            description="A description of the column. This is optional but recommended."
        ),
    ] = None

    database_params: Annotated[
        DatabaseParamsModel,
        Field(description="Database constraints and other parameters."),
    ] = DatabaseParamsModel()

    is_source_column: Annotated[
        bool,
        Field(
            description="Whether this is a column is part of the source catalogue or "
            "was added during the xmatch process, such as HEALPix indices."
        ),
    ] = True

    source_catalogue_params: Annotated[
        SourceCatalogueParamsColumnModel | None,
        Field(description="Parameters of the column in the original source data."),
    ] = None


class Schema:
    """Class representing a schema for a catalogue.

    Parameters
    ----------
    schema
        The schema for the catalogue. Either a path to the schema file or a
        dictionary representing the schema.

    """

    def __init__(self, schema: str | pathlib.Path | dict) -> None:
        if isinstance(schema, (str, pathlib.Path)):
            self._raw_schema, self.schema_file = self.read_schema(schema)
        elif isinstance(schema, dict):
            self._raw_schema = schema
            self.schema_file = None
        else:
            raise ValueError("Schema must be a path to a schema file or a dictionary")

        self.schema: SchemaModel = SchemaModel(**self._raw_schema)

    def get_ra_dec_columns(self) -> tuple[str, str]:
        """Returns the names of the RA and Dec columns in the schema.

        Returns
        -------
        ra_column
            The name of the RA column.
        dec_column
            The name of the Dec column.

        """

        ra_column = self.schema.ra_column
        dec_column = self.schema.dec_column

        if ra_column is None:
            if "ra" in self.schema.columns:
                ra_column = "ra"
                warnings.warn(
                    "RA column not specified in schema, "
                    "but 'ra' column exists. Using 'ra' as the RA column.",
                    UserWarning,
                )
            else:
                raise ValueError("RA column not specified in schema.")

        if dec_column is None:
            if "dec" in self.schema.columns:
                dec_column = "dec"
                warnings.warn(
                    "Dec column not specified in schema, "
                    "but 'dec' column exists. Using 'dec' as the Dec column.",
                    UserWarning,
                )
            else:
                raise ValueError("Dec column not specified in schema.")

        return ra_column, dec_column

    def read_schema(
        self,
        schema_file: str | pathlib.Path,
    ) -> tuple[dict, pathlib.Path]:
        """Returns the schema for the provided schema file.

        Parameters
        ----------
        schema_file
            The path to the schema file. It can also be the name of a schema in the
            ``schemas/`` directory.

        Returns
        -------
        schema
            The schema as a dictionary of column names and Polars data types.
        schema_file
            The path to the schema file that was read.

        """

        if isinstance(schema_file, str) and not pathlib.Path(schema_file).is_file():
            schema_path = XMATCH_RS_ROOT / "schemas" / f"{schema_file}.yaml"
        else:
            schema_path = pathlib.Path(schema_file)

        if not schema_path.is_file():
            raise FileNotFoundError(f"Schema file '{schema_path}' does not exist")

        schema_data = read_yaml_file(schema_path, return_class=dict)

        return schema_data, schema_path

    def write_schema(self, path: str | pathlib.Path) -> None:
        """Writes the schema to a YAML file.

        Parameters
        ----------
        path
            The path where the schema file will be written.

        """

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self._raw_schema, f, sort_keys=False, indent=2)

    def to_dict(self, only_source_columns: bool = False) -> dict:
        """Returns the schema as a dictionary.

        Parameters
        ----------
        only_source_columns
            If ``True``, only returns the columns that are part of the source
            catalogue.

        """

        schema = self.schema
        if only_source_columns:
            schema = schema.get_source_columns()

        return schema.model_dump()

    def get_primary_keys(self) -> list[str]:
        """Returns a list of the columns that are marked as primary keys."""

        return [
            column_name
            for column_name, column_data in self.schema.columns.items()
            if column_data.database_params.primary_key
        ]

    def get_polars_schema(
        self,
        source_columns: bool = False,
    ) -> dict[str, polars.DataType]:
        """Returns the schema as a dictionary of column names and Polars data types.

        Parameters
        ----------
        source_columns
            If ``True``, only returns the columns that are part of the source
            catalogue. The data types of these columns are taken from the
            ``source_catalogue.dtype`` field if present, otherwise the ``dtype``
            field.

        """

        schema = self.schema
        if source_columns:
            schema = schema.get_source_columns()

            pl_sch: dict[str, polars.DataType] = {}

            for c_name, c_data in schema.columns.items():
                if c_data.is_source_column:
                    if c_data.source_catalogue_params:
                        cat_params = c_data.source_catalogue_params
                        cat_name_source = cat_params.column_name or c_name
                        dtype_source = cat_params.dtype or c_data.dtype
                        polars_dtype = get_polars_dtype_from_string(dtype_source)
                        pl_sch[cat_name_source] = polars_dtype
                    else:
                        polars_dtype = get_polars_dtype_from_string(c_data.dtype)
                        pl_sch[c_name] = polars_dtype

            return pl_sch

        return {
            column_name: get_polars_dtype_from_string(column_data.dtype)
            for column_name, column_data in schema.columns.items()
        }
