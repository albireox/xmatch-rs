#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-03-17
# @Filename: schema.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import polars

from sdsstools import read_yaml_file

import xmatch_rs


__all__: list[str] = ["get_schema", "XMATCH_RS_ROOT"]


XMATCH_RS_ROOT = pathlib.Path(xmatch_rs.__file__).parent


def get_polars_dtype_from_string(dtype_str: str):
    """Retrieves a Polars DataType object from its string representation.

    Parameters
    ----------
        dtype_str
        The name of the Polars data type as a string (e.g., "Int64", "Utf8", "Date").

    Returns
    -------
    polars.DataType
        A Polars DataType object.

    """

    try:
        return getattr(polars, dtype_str)
    except AttributeError:
        raise ValueError(f"'{dtype_str}' is not a valid Polars data type")


def get_schema(schema_file: str | pathlib.Path) -> dict[str, polars.DataType]:
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

    """

    if isinstance(schema_file, str) and not pathlib.Path(schema_file).is_file():
        schema_path = XMATCH_RS_ROOT / "schemas" / f"{schema_file}.yaml"
    else:
        schema_path = pathlib.Path(schema_file)

    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema file '{schema_path}' does not exist")

    schema_data = read_yaml_file(schema_path, return_class=dict)
    if "schema" not in schema_data:
        raise ValueError(f"Schema file '{schema_path}' does not contain a 'schema' key")

    schema: dict[str, polars.DataType] = {}
    for column_name, column_data in schema_data["schema"].items():
        dtype_str = column_data["dtype"]
        schema[column_name] = get_polars_dtype_from_string(dtype_str)

    return schema
