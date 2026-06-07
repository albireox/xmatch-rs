#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-06-06
# @Filename: create.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import TYPE_CHECKING


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


def create_table_sql(hats: HATS) -> str:
    """Generates a SQL statement to create a table.

    Parameters
    ----------
    hats
        The HATS dataset for which to create the table.

    Returns
    -------
    str
        The SQL statement to create the table.

    """

    columns = hats.schema.get_columns(use_database_name=True)

    schema = hats.schema.model.database.schema_
    table_name = hats.schema.model.database.table_name

    sql = f"CREATE TABLE {schema}.{table_name} (\n"

    for column_name, column in columns.items():
        sql += f"  {column_name} {DTYPE_TO_SQL.get(column.dtype, column.dtype)},\n"

    sql = sql.rstrip(",\n") + "\n);"

    return sql
