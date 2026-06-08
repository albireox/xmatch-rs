#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-03-17
# @Filename: schema.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import enum
import pathlib

from typing import Annotated, Any, Type

import polars
import yaml
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, field_validator

from sdsstools import read_yaml_file

from xmatch_rs.tools import get_polars_dtype_from_string


__all__: list[str] = ["SchemaModel", "Schema"]


class EpochFormat(str, enum.Enum):
    """Enumeration of possible epoch formats."""

    JULIAN_YEAR = "jyear"
    JD = "jd"


class SourceCatalogueFormat(str, enum.Enum):
    """Enumeration of possible formats for the original source catalogue."""

    CSV = "csv"
    PARQUET = "parquet"


class Dtype(str, enum.Enum):
    """Enumeration of possible data types for columns in the schema."""

    STRING = "String"
    UINT8 = "UInt8"
    UINT16 = "UInt16"
    UINT32 = "UInt32"
    UINT64 = "UInt64"
    INT8 = "Int8"
    INT16 = "Int16"
    INT32 = "Int32"
    INT64 = "Int64"
    FLOAT16 = "Float16"
    FLOAT32 = "Float32"
    FLOAT64 = "Float64"
    BOOLEAN = "Boolean"


TableColumn = Annotated[
    str,
    Field(description="A table_name.column value.", pattern="^[^.]+\\.[^.]+$"),
]


class SchemaModel(BaseModel):
    """Model representing a schema file."""

    catalogue_name: Annotated[
        str,
        Field(description="Short version of the name of the catalogue."),
    ]

    url: Annotated[
        AnyUrl,
        Field(description="A URL with catalogue information."),
    ]

    bib_reference: Annotated[
        str | None,
        Field(description="A bibliographic reference for the catalogue."),
    ] = None

    bib_reference_url: Annotated[
        AnyUrl | None,
        Field(description="A URL to a bibliographic reference for the catalogue."),
    ] = None

    comments: Annotated[
        str | list[str] | None,
        Field(description="Any comments about the catalogue."),
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
        bool | None,
        Field(
            description="Whether the proper motion in RA column has already been "
            "deprojected by multiplying by cos(dec)."
        ),
    ] = None

    pmdec_column: Annotated[
        str | None,
        Field(description="The name of the proper motion in Dec column."),
    ] = None

    epoch: Annotated[
        float | None,
        Field(description="The epoch of the coordinates as Julian year."),
    ] = None

    epoch_format: Annotated[
        EpochFormat | None,
        Field(description="The format of the epoch."),
    ] = None

    epoch_column: Annotated[
        str | None,
        Field(description="The column containing the epoch of the coordinates."),
    ] = None

    source_catalogue: Annotated[
        SourceCatalogueModel,
        Field(description="Source data information."),
    ]

    extra_columns: Annotated[
        dict[str, ColumnModel],
        Field(
            description="A dictionary of extra column names and their parameters.",
            default_factory=dict,
        ),
    ]

    database: Annotated[DatabaseModel, Field(description="Database parameters.")]

    @field_validator("extra_columns", mode="after")
    @classmethod
    def validate_extra_columns(
        cls,
        extra_columns: dict[str, ColumnModel],
    ) -> dict[str, ColumnModel]:
        for column_data in extra_columns.values():
            for field in ["rename_to", "cast_to"]:
                if getattr(column_data, field) is not None:
                    raise ValueError(
                        f"The {field!r} field is not allowed for extra columns."
                    )
            if column_data.apply_functions != []:
                raise ValueError(
                    "The 'apply_functions' field is not allowed for extra columns."
                )

        return extra_columns


class SourceCatalogueModel(BaseModel):
    """Model representing the parameters to read the original source data."""

    data_url: Annotated[
        AnyUrl,
        Field(description="A URL to the original source of the data."),
    ]

    format: Annotated[
        SourceCatalogueFormat,
        Field(description="The format of the original source data."),
    ]

    read_params: Annotated[
        ReadParamsModel,
        Field(
            description="Additional parameters to read the original source data. "
            "The specific parameters depend on the format of the source data.",
            default_factory=ReadParamsModel,
        ),
    ]

    columns: Annotated[
        dict[str, ColumnModel],
        Field(
            description="A dictionary of column names in the catalogue and their "
            "parameters in the original source data.",
            default_factory=dict,
        ),
    ]


class ReadParamsModel(BaseModel):
    """Model representing additional parameters to read the original source data."""

    csv_has_header: Annotated[
        bool | None,
        Field(description="Whether the CSV file has a header row."),
    ] = None

    csv_delimiter: Annotated[
        str | None,
        Field(description="The delimiter used in the CSV file."),
    ] = None

    csv_comment_prefix: Annotated[
        str | None,
        Field(description="The prefix used for comment lines in the CSV file."),
    ] = None

    csv_null_values: Annotated[
        list[str],
        Field(
            description="A list of strings that should be interpreted as null "
            "values for CSV files.",
            default_factory=list,
        ),
    ]

    nan_to_null: Annotated[
        bool,
        Field(
            description="Whether to convert NaN values to null values when reading "
            "the source data."
        ),
    ] = False


class ColumnModel(BaseModel):
    """Model representing a column in the schema."""

    dtype: Annotated[
        Dtype,
        Field(
            description="The data type of the column. This should be a string "
            "that can be converted to a Polars data type."
        ),
    ]

    unit: Annotated[
        str | None,
        Field(description="The unit of the column."),
    ] = None

    description: Annotated[
        str | None,
        Field(description="A description of the column."),
    ] = None

    cast_to: Annotated[
        Dtype | None,
        Field(
            description="If the column in the original source data needs to be cast "
            "to a different data type, this field specifies the target data type. "
            "Must be a string that can be converted to a Polars data type."
        ),
    ] = None

    rename_to: Annotated[
        str | None,
        Field(
            description="Rename the column to this name after reading it "
            "from the original source data."
        ),
    ] = None

    database_name: Annotated[
        str | None,
        Field(description="The name of the column in the database."),
    ] = None

    apply_functions: Annotated[
        list[str],
        Field(
            description="A list of functions to apply to the column after reading it "
            "from the original source data. The functions are applied in the order "
            "they are listed. See the xmatch_rs.convert.callbacks module for available "
            "functions.",
            default_factory=list,
        ),
    ]


class DatabaseModel(BaseModel):
    """Model representing database parameters for the catalogue."""

    table_name: Annotated[
        str,
        Field(description="The database table where the catalogue will be loaded."),
    ]

    schema_: Annotated[
        str,
        Field(
            alias="schema",
            serialization_alias="schema",
            description="The database schema where the catalogue will be loaded.",
        ),
    ]

    primary_key: Annotated[
        str,
        Field(description="The column that will be used as the primary key."),
    ]

    indices: Annotated[
        list[str | list[str]],
        Field(description="A list of columns to be indexed.", default_factory=list),
    ]

    foreign_keys: Annotated[
        list[tuple[str, TableColumn]],
        Field(
            description="A list of columns that are foreign keys to other catalogues. "
            "the format must be a list of tuples of the form "
            "(column, other_catalogue.column).",
            default_factory=list,
        ),
    ]

    model_config = ConfigDict(serialize_by_alias=True)


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
            self._raw_schema, self.schema_file = self._read_schema(schema)
        elif isinstance(schema, dict):
            self._raw_schema = schema
            self.schema_file = None
        else:
            raise ValueError("Schema must be a path to a schema file or a dictionary")

        self.model = SchemaModel(**self._raw_schema)

    def get_columns(
        self,
        only_source_columns: bool = False,
        apply_rename_to: bool = True,
        use_database_name: bool = False,
        apply_cast_to: bool = True,
    ) -> dict[str, ColumnModel]:
        """Returns a dictionary of column names and their parameters."""

        columns = self.model.source_catalogue.columns.copy()

        if not only_source_columns:
            columns.update(self.model.extra_columns.copy())

        new_columns: dict[str, ColumnModel] = {}
        for column_name, column in columns.items():
            new_column_name = column_name

            if use_database_name and column.database_name is not None:
                new_column_name = column.database_name
            elif column.rename_to is not None and apply_rename_to:
                new_column_name = column.rename_to

            new_columns[new_column_name] = column

            if apply_cast_to and column.cast_to is not None:
                new_col = column.model_copy(update={"dtype": column.cast_to})
                new_col.cast_to = None
                new_columns[new_column_name] = new_col

        return new_columns

    def get_column(
        self,
        column_name: str,
        apply_rename_to: bool = True,
        use_database_name: bool = False,
        apply_cast_to: bool = True,
    ) -> ColumnModel | None:
        """Returns the model for a specific column.

        Parameters
        ----------
        column_name
            The name of the column.
        apply_rename_to
            Whether to apply the renaming specified in the schema to the column names.
        use_database_name
            Whether to use the database column names instead of the original
            column names.
        apply_cast_to
            Whether to apply the casting specified in the schema to the column
            data types.

        """

        columns = self.get_columns(
            apply_rename_to=apply_rename_to,
            apply_cast_to=apply_cast_to,
            use_database_name=use_database_name,
        )
        if column_name not in columns:
            return None

        return columns[column_name]

    def get_ra_dec_columns(self) -> tuple[str, str]:
        """Returns the names of the RA and Dec columns in the schema.

        Returns
        -------
        ra_column
            The name of the RA column.
        dec_column
            The name of the Dec column.

        """

        ra_column = self.model.ra_column
        dec_column = self.model.dec_column

        if ra_column is None:
            raise ValueError("RA column not specified in schema.")

        if dec_column is None:
            raise ValueError("Dec column not specified in schema.")

        return ra_column, dec_column

    def _read_schema(
        self,
        schema_file: str | pathlib.Path,
    ) -> tuple[dict[str, Any], pathlib.Path]:
        """Returns the schema for the provided schema file.

        Parameters
        ----------
        schema_file
            The path to the schema file.

        Returns
        -------
        schema
            The schema as a dictionary of column names and Polars data types.
        schema_file
            The path to the schema file that was read.

        """

        schema_path = pathlib.Path(schema_file)

        if not schema_path.is_file() or not schema_path.exists():
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
            yaml.dump(self.model.model_dump(mode="json"), f, sort_keys=False, indent=2)

    def to_dict(self) -> dict:
        """Returns the schema as a dictionary.

        Parameters
        ----------
        only_source_columns
            If ``True``, only returns the columns that are part of the source
            catalogue.

        """

        model = self.model.model_copy()

        return model.model_dump(mode="json")

    def get_primary_key(self) -> str:
        """Returns the primary key column name."""

        return self.model.database.primary_key

    def get_polars_schema(self, source_catalogue: bool = False) -> polars.Schema:
        """Returns the schema as a Polars Schema.

        Parameters
        ----------
        source_catalogue
            If ``True``, only returns the columns that are part of the source
            catalogue. The data types of these columns are taken from the
            ``source_catalogue.columns.dtype`` fields ignoring any subsequent casting.

        """

        columns = self.get_columns(only_source_columns=source_catalogue)

        schema: dict[str, Type[polars.DataType]] = {}

        for column_name, column in columns.items():
            dtype_str = column.dtype.value

            if not source_catalogue:
                if column.cast_to is not None:
                    dtype_str = column.cast_to.value
                if column.rename_to is not None:
                    column_name = column.rename_to

            schema[column_name] = get_polars_dtype_from_string(dtype_str)

        return polars.Schema(schema)
