#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-07-21
# @Filename: tools.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import polars

from xmatch_rs.expressions import get_healpix


def add_healpix_29_column(
    database_uri: str,
    table_name: str,
    primary_key: str,
    ra_column: str = "ra",
    dec_column: str = "dec",
    column_name: str = "healpix_29",
    temporary_schema: str = "sandbox",
    chunk_size: int | None = 10_000_000,
):
    """Adds a ``healpix_29`` column to a table in the database.

    Adds a new column ``healpix_29`` to the specified table with the HEALPix element
    of order 29 for the given ``ra_column`` and ``dec_column`` coordinates.

    Parameters
    ----------
    database_uri
        The URI of the database to connect to.
    table_name
        The name of the table to add the column to. Can be schema-qualified.
    primary_key
        The name of the primary key column in the table. Must be sortable.
    ra_column
        The name of the column containing right ascension values.
    dec_column
        The name of the column containing declination values.
    column_name
        The name of the new column to create.
    temporary_schema
        The name of the temporary schema to use for intermediate operations.
    chunk_size
        The size of chunks to process at a time. If None, all rows are processed
        at once.

    """

    if "." in table_name:
        schema, table_name = table_name.split(".", 1)
    else:
        schema = None

    tmp_table = f'"{temporary_schema}"."{table_name}_healpix_29"'

    schema_table = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'

    offset = 0
    first = True
    n_rows = 0
    while True:
        query = f"""
        SELECT {primary_key}, {ra_column}, {dec_column}
        FROM {schema_table}
        ORDER BY {primary_key}
        LIMIT {chunk_size}
        """

        if chunk_size is not None:
            query += f" OFFSET {offset}"

        data = polars.read_database_uri(query, uri=database_uri, engine="adbc")
        if data.is_empty():
            break

        data = (
            data.with_columns(
                get_healpix(
                    polars.col(ra_column),
                    polars.col(dec_column),
                    order=29,
                ).alias(column_name)
            )
            .select(polars.col(primary_key), polars.col(column_name))
            .cast({column_name: polars.Int64})
        )

        data.write_database(
            table_name=tmp_table,
            connection=database_uri,
            engine="adbc",
            if_table_exists="fail"
            if first
            else "append",  # See https://github.com/pola-rs/polars/issues/26906
        )

        if chunk_size is not None:
            offset += chunk_size

        n_rows += len(data)
        print(f"Processed {n_rows:,} rows.", end="\r")

        first = False

    print()
