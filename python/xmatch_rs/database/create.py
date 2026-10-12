#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-06-06
# @Filename: create.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import TYPE_CHECKING

import polars


if TYPE_CHECKING:
    from xmatch_rs import HATS


__all__ = ["create_table_sql", "DTYPE_TO_SQL"]


DTYPE_TO_SQL = {
    "UInt8": "SMALLINT",
    "UInt16": "INTEGER",
    "UInt32": "BIGINT",
    "UInt64": "BIGINT",
    "Int8": "SMALLINT",
    "Int16": "SMALLINT",
    "Int32": "INTEGER",
    "Int64": "BIGINT",
    "Float16": "REAL",
    "Float32": "REAL",
    "Float64": "DOUBLE PRECISION",
    "String": "TEXT",
    "Boolean": "BOOLEAN",
}


def create_table_sql(
    hats: HATS | None = None,
    frame: polars.DataFrame | polars.LazyFrame | str | pathlib.Path | None = None,
    table_name: str | None = None,
    schema: str | None = None,
) -> str:
    """Generates a SQL statement to create a table.

    Parameters
    ----------
    hats
        The HATS dataset for which to create the table.
    frame
        The Polars DataFrame or LazyFrame to use for generating the table schema.
        Can also be a path to a Parquet file.

    Returns
    -------
    str
        The SQL statement to create the table.

    """

    if hats is None and frame is None:
        raise ValueError("Either hats or frame must be provided.")

    if hats is not None:
        columns = hats.schema.get_polars_schema(use_database_name=True)
        schema = hats.schema.model.database.schema_
        table_name = hats.schema.model.database.table_name
    else:
        if schema is None or table_name is None:
            raise ValueError("'schema' and 'table_name' must be provided with 'frame'.")

        if isinstance(frame, (str, pathlib.Path)):
            frame = polars.scan_parquet(frame)

        if isinstance(frame, polars.DataFrame):
            columns = dict(frame.schema)
        elif isinstance(frame, polars.LazyFrame):
            columns = dict(frame.collect_schema())
        else:
            raise ValueError("Unsupported frame type.")

    sql = f"CREATE TABLE {schema}.{table_name} (\n"

    for column_name, column_dtype in columns.items():
        if isinstance(column_dtype, (polars.Array, polars.List)):
            inner = column_dtype.inner
            if (inner_dtype_str := str(inner)) in DTYPE_TO_SQL:
                sql_type = DTYPE_TO_SQL[inner_dtype_str]
                if isinstance(column_dtype, polars.Array):
                    size = column_dtype.size
                    sql += f"  {column_name} {sql_type}[{size}],\n"
                elif isinstance(column_dtype, polars.List):
                    sql += f"  {column_name} {sql_type}[],\n"
            else:
                raise ValueError(f"Unsupported array/list column {column_name!r}.")
        else:
            sql_type = DTYPE_TO_SQL[str(column_dtype)]
            sql += f"  {column_name} {sql_type},\n"

    sql = sql.rstrip(",\n") + "\n);"

    return sql
