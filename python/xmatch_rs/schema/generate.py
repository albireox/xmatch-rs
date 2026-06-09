#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-06-03
# @Filename: generate.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import Any, Literal

import polars
import questionary
from questionary import ValidationError, Validator

from xmatch_rs import console

from .schema import SchemaModel


__all__ = ["generate_schema"]


class NameValidator(Validator):
    def validate(self, document):
        if len(document.text) == 0:
            raise ValidationError(
                message="Please enter a value",
                cursor_position=len(document.text),
            )


def infer_data_format(
    data_path: str | pathlib.Path,
    csv_has_header: bool = True,
    csv_delimiter: str = ",",
    csv_comment_prefix: str = "#",
    csv_null_values: list[str] = [],
) -> Literal["csv", "parquet", "fits"] | None:
    """Infers the data format from the file extension.

    Parameters
    ----------
    data_path
        Path to the sample dataset.
    csv_has_header
        For CSV files, whether the first row contains column names.
    csv_delimiter
        For CSV files, the delimiter to use.
    csv_comment_prefix
        For CSV files, the comment prefix to use.
    csv_null_values
        For CSV files, a list of strings that should be interpreted as null values.

    Returns
    -------
    Literal["csv", "parquet", "fits"] | None
        The inferred data format, or None if it cannot be inferred.

    """

    data_path = pathlib.Path(data_path)

    if data_path.is_file():
        name = pathlib.Path(data_path).name.lower()

        if name.endswith(".csv") or name.endswith(".csv.gz"):
            return "csv"
        elif name.endswith(".parquet"):
            return "parquet"
        elif name.endswith(".fits") or name.endswith(".fit"):
            return "fits"
        elif name.endswith(".fits.gz") or name.endswith(".fit.gz"):
            return "fits"
        else:
            return None

    else:
        try:
            polars.scan_csv(
                data_path,
                has_header=csv_has_header,
                separator=csv_delimiter,
                comment_prefix=csv_comment_prefix,
                null_values=csv_null_values,
            ).head(1).collect()
            return "csv"
        except Exception:
            pass

        try:
            polars.scan_parquet(data_path).head(1).collect()
            return "parquet"
        except Exception:
            pass

        return None


def generate_schema(
    data_path: str | pathlib.Path | None = None,
    format: Literal["csv", "parquet", "fits"] | None = None,
    skip_optional: bool = False,
    csv_has_header: bool = True,
    csv_delimiter: str = ",",
    csv_comment_prefix: str = "#",
    csv_null_values: list[str] = [],
) -> SchemaModel | None:
    """Interactively generates a schema from a sample data file.

    Parameters
    ----------
    data_path
        Path to the sample data file. Used to infer column names and types.
    format
        The format of the sample data file. If not provided, it will be inferred
        from the file extension.
    skip_optional
        If `True`, optional fields will be skipped and set to the default value.
    csv_has_header
        For CSV files, whether the first row contains column names.
    csv_delimiter
        For CSV files, the delimiter to use.
    csv_comment_prefix
        For CSV files, the comment prefix to use.
    csv_null_values
        For CSV files, a list of strings that should be interpreted as null values.

    Returns
    -------
    SchemaModel
        The generated schema model.

    """

    from xmatch_rs.convert import read_fits_to_dataframe

    data_path = pathlib.Path(data_path) if data_path is not None else None

    if data_path and not format:
        format = infer_data_format(
            data_path,
            csv_has_header=csv_has_header,
            csv_delimiter=csv_delimiter,
            csv_comment_prefix=csv_comment_prefix,
            csv_null_values=csv_null_values,
        )

    # Required catalogue information.
    catalogue_name: str = questionary.text(
        "Catalogue name\n",
        qmark="🏷️",
        validate=NameValidator,
    ).unsafe_ask()

    url: str = questionary.text(
        "URL of the catalogue\n",
        qmark="🔗",
        validate=NameValidator,
    ).unsafe_ask()

    data_url: str = questionary.text(
        "URL of the source data\n",
        qmark="💾",
        validate=NameValidator,
    ).unsafe_ask()

    # Optional catalogue information.
    if not skip_optional:
        bib_reference: str | None = (
            questionary.text(
                "Bibliographic reference\n",
                qmark="📚",
            ).unsafe_ask()
            or None
        )

        bib_reference_url: str | None = (
            questionary.text(
                "URL of the bibliographic reference\n",
                qmark="🔗",
            ).unsafe_ask()
            or None
        )

        comments: str | None = (
            questionary.text(
                "Comments\n",
                qmark="💬",
            ).unsafe_ask()
            or None
        )
    else:
        bib_reference: str | None = None
        bib_reference_url: str | None = None
        comments: str | None = None

    # Try to read the sample data, if specified.
    source_columns: dict[str, dict[str, Any]] = {}
    if data_path:
        if format is None:
            format: str = questionary.select(
                "format of the sample dataset\n",
                choices=[
                    questionary.Choice("CSV", value="csv"),
                    questionary.Choice("Parquet", value="parquet"),
                    questionary.Choice("FITS", value="fits"),
                ],
            ).unsafe_ask()

        try:
            if format == "csv":
                df = polars.read_csv(
                    data_path,
                    n_rows=1000,
                    has_header=csv_has_header,
                    separator=csv_delimiter,
                    comment_prefix=csv_comment_prefix,
                    null_values=csv_null_values,
                )
            elif format == "parquet":
                df = polars.read_parquet(data_path, n_rows=1000)
            elif format == "fits":
                df = read_fits_to_dataframe(data_path).head(10)
            else:
                console.print(f"[red]Unsupported format {format!r}.[/]")
                return None

            pl_schema = df.schema
            for column, dtype in pl_schema.items():
                source_columns[column] = {
                    "dtype": str(dtype),
                    "unit": None,
                    "description": None,
                }

        except Exception as err:
            console.print(f"[red]Error reading data: {err!s}[/]")
            console.print("[yellow]Column information will not be available.[/]")

    # Read parameters
    if format == "csv":
        read_params = {
            "csv_has_header": csv_has_header,
            "csv_delimiter": csv_delimiter,
            "csv_comment_prefix": csv_comment_prefix,
        }
    else:
        read_params = {}

    # Primary key.
    primary_key: str = questionary.text(
        "Primary key column\n",
        qmark="🔑",
        validate=NameValidator,
    ).unsafe_ask()

    create_primary_key: bool = False
    if primary_key not in source_columns:
        create_primary_key: bool = questionary.confirm(
            f"Primary key column {primary_key!r} not found in the sample data. "
            "Create it as an auto-incrementing integer?\n",
            qmark="🔑",
        ).unsafe_ask()

    # Coordinate and epoch information.
    if not skip_optional:
        ra_column: str | None = (
            questionary.text(
                "Right ascension column\n",
                qmark="🧭",
                default="ra" if "ra" in source_columns else "",
            ).unsafe_ask()
            or None
        )

        ra_column_unit: str | None = questionary.select(
            "Right ascension column unit\n",
            qmark="🧭",
            default="degrees",
            choices=[
                questionary.Choice("Degrees", value="degrees"),
                questionary.Choice("Hours", value="hours"),
                questionary.Choice("Radians", value="radians"),
            ],
        ).unsafe_ask()

        dec_column: str | None = (
            questionary.text(
                "Declination column\n",
                qmark="🧭",
                default="dec" if "dec" in source_columns else "",
            ).unsafe_ask()
            or None
        )

        dec_column_unit: str | None = questionary.select(
            "Declination column unit\n",
            qmark="🧭",
            default="degrees",
            choices=[
                questionary.Choice("Degrees", value="degrees"),
                questionary.Choice("Hours", value="hours"),
            ],
        ).unsafe_ask()

        pmra_column: str | None = (
            questionary.text(
                "Proper motion in right ascension column\n",
                qmark="🧭",
                default="pmra" if "pmra" in source_columns else "",
            ).unsafe_ask()
            or None
        )

        pmdec_column: str | None = (
            questionary.text(
                "Proper motion in declination column\n",
                qmark="🧭",
                default="pmdec" if "pmdec" in source_columns else "",
            ).unsafe_ask()
            or None
        )

        pmra_is_cosdec: bool = questionary.select(
            "Is the proper motion in right ascension column as true angle?\n",
            qmark="🧭",
            choices=[
                questionary.Choice("Yes", value=True),
                questionary.Choice("No", value=False),
            ],
        ).unsafe_ask()

        epoch: str | None = (
            questionary.text(
                "Epoch of the coordinates\n",
                qmark="📅",
            ).unsafe_ask()
            or None
        )

        if not epoch:
            epoch_column: str | None = (
                questionary.text(
                    "Epoch column\n",
                    qmark="📅",
                ).unsafe_ask()
                or None
            )
        else:
            epoch_column = None

        epoch_format: str = questionary.select(
            "Epoch format\n",
            qmark="📅",
            choices=[
                questionary.Choice("Julian year", value="jyear"),
                questionary.Choice("JD", value="JD"),
            ],
        ).unsafe_ask()

    else:
        ra_column: str | None = None
        dec_column: str | None = None
        ra_column_unit: str | None = None
        dec_column_unit: str | None = None
        pmra_column: str | None = None
        pmdec_column: str | None = None
        pmra_is_cosdec: bool = True
        epoch: str | None = None
        epoch_column: str | None = None
        epoch_format: str | None = None

    # Add HATS extra columns.
    if not skip_optional:
        add_hats_columns: bool = questionary.confirm(
            "Add HATS extra columns (_healpy_29, etc.)?\n",
            qmark="✨",
        ).unsafe_ask()
        if add_hats_columns:
            extra_columns = {
                "_healpy_29": {
                    "dtype": "UInt64",
                    "unit": None,
                    "description": "HEALPix index at depth 29",
                    "database_name": "healpix_29",
                },
                "Norder": {
                    "dtype": "Int8",
                    "description": "HEALPix order of the target in the HATS partition",
                    "database_name": "hats_norder",
                },
                "Dir": {
                    "dtype": "UInt32",
                    "description": "HATS directory grouping",
                    "database_name": "hats_dir",
                },
                "Npix": {
                    "dtype": "UInt32",
                    "description": "HEALPix pixel of order norder in which this target "
                    "is included in the HATS partition.",
                    "database_name": "hats_npix",
                },
            }
    else:
        extra_columns: dict[str, dict[str, str | None]] = {}

    if primary_key not in source_columns and create_primary_key:
        extra_columns[primary_key] = {
            "dtype": "UInt64",
            "unit": None,
            "description": "Auto-incrementing primary key",
        }

    # Database questions.
    table_name: str = questionary.text(
        "Database table name\n",
        qmark="🗄️",
        default=catalogue_name.lower(),
        validate=NameValidator,
    ).unsafe_ask()

    db_schema: str = questionary.text(
        "Database schema\n",
        qmark="🗄️",
        default="catalogdb",
        validate=NameValidator,
    ).unsafe_ask()

    if not skip_optional:
        indices = questionary.text(
            "Database indices (space-separated, optional)\n",
            qmark="🗄️",
        ).unsafe_ask()
    else:
        indices: str | None = None

    schema = SchemaModel.model_validate(
        {
            "catalogue_name": catalogue_name,
            "url": url,
            "bib_reference": bib_reference,
            "bib_reference_url": bib_reference_url,
            "comments": comments,
            "primary_key": primary_key,
            "create_primary_key": create_primary_key,
            "ra_column": ra_column,
            "dec_column": dec_column,
            "ra_column_unit": ra_column_unit,
            "dec_column_unit": dec_column_unit,
            "pmra_column": pmra_column,
            "pmdec_column": pmdec_column,
            "pmra_is_cosdec": pmra_is_cosdec,
            "epoch": float(epoch) if epoch else None,
            "epoch_column": epoch_column,
            "epoch_format": epoch_format,
            "source_catalogue": {
                "data_url": data_url,
                "read_params": read_params,
                "format": format,
                "columns": source_columns,
            },
            "extra_columns": extra_columns,
            "database": {
                "table_name": table_name,
                "schema": db_schema,
                "indices": indices.strip().split() if indices else [],
            },
        }
    )

    return schema
