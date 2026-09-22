#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-07-21
# @Filename: tools.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import datetime
import pathlib
import re
import subprocess

from typing import Literal, NamedTuple

import polars
import psycopg
import psycopg.sql as sql
from psycopg.sql import Identifier
from rich.console import Console

from sdsstools.utils import Timer

from xmatch_rs.expressions import get_healpix


def table_exists(database_uri: str, table_name: str, schema: str | None = None) -> bool:
    """Check if a table exists in the database."""

    db_conn = psycopg.connect(database_uri)
    with db_conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = %s
                AND table_schema = %s
            );
            """,
            (table_name, schema or "public"),
        )

        if (result := cursor.fetchone()) is None:
            return False

        return result[0]


def add_healpix_29_column(
    database_uri: str,
    table_name: str,
    primary_key: str,
    ra_column: str = "ra",
    dec_column: str = "dec",
    column_name: str = "healpix_29",
    temp_schema: str = "sandbox",
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
    temp_schema
        The name of the temporary schema to use for intermediate operations.
    chunk_size
        The size of chunks to process at a time. If None, all rows are processed
        at once.

    """

    if "." in table_name:
        schema, table_name = table_name.split(".", 1)
    else:
        schema = "public"

    schema_table = f"{schema}.{table_name}" if schema else f"{table_name}"

    if not table_exists(database_uri, table_name, schema):
        raise ValueError(f"Table {schema_table} does not exist")

    tmp_table = f"{temp_schema}.{table_name}_healpix_29"

    if table_exists(database_uri, table_name, schema):
        raise ValueError(f"Temporary table {tmp_table} already exists")

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
        print(f"\rProcessed {n_rows:,} rows.", end="", flush=True)

        first = False

    print()

    with psycopg.connect(database_uri) as conn:
        with conn.cursor() as cur:
            # print(f"Adding primary key to temporary table {tmp_table} ...")
            tmp_id = Identifier(temp_schema, f"{table_name}_healpix_29")
            # cur.execute(t"ALTER TABLE {tmp_id:i} ADD PRIMARY KEY ({primary_key:i})")
            # cur.execute(t"ANALYZE {tmp_id:i}")

            # print("Adding column to original table...")
            table_id = Identifier(schema, table_name)
            # cur.execute(t"ALTER TABLE {table_id:i} ADD COLUMN {column_name:i} BIGINT")

            print("Updating original table with HEALPix values...")
            cur.execute(
                t"UPDATE {table_id:i} SET {column_name:i} = "
                t"(SELECT {column_name:i} FROM {tmp_id:i} WHERE "
                t"{tmp_id:i}.{primary_key:i} = {table_id:i}.{primary_key:i})"
            )

            print("Adding index to original table ...")
            cur.execute(t"CREATE INDEX ON {table_id:i} ({column_name:i})")


def get_table_columns(
    database_uri: str,
    table_name: str,
    schema: str = "public",
) -> list[tuple[str, str, bool]]:
    """Returns a list of columns in a table.

    Parameters
    ----------
    database_uri
        The URI of the database to connect to.
    table_name
        The name of the table to get the columns from.
    schema
        The schema of the table.

    Returns
    -------
    A list of tuples with the column name, data type, and whether the column is
    nullable.

    """

    with psycopg.connect(database_uri) as src_conn, src_conn.cursor() as src_cursor:
        src_cursor.execute(
            """
            SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s
                AND c.relname = %s
                AND a.attnum > 0
                AND NOT a.attisdropped
            ORDER BY a.attnum
            """,
            (schema, table_name),
        )
        columns = src_cursor.fetchall()

    return columns


def get_primary_key_columns(
    database_uri: str,
    table_name: str,
    schema: str = "public",
) -> list[str]:
    """Returns a list of primary key columns in a table.

    Parameters
    ----------
    database_uri
        The URI of the database to connect to.
    table_name
        The name of the table to get the primary key columns from.
    schema
        The schema of the table.

    Returns
    -------
    A list of primary key column names.

    """

    with psycopg.connect(database_uri) as src_conn, src_conn.cursor() as src_cursor:
        src_cursor.execute(
            """
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a
                ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = %s::regclass
                AND i.indisprimary
            ORDER BY array_position(i.indkey, a.attnum)
            """,
            (f"{schema}.{table_name}",),
        )
        pk_columns = [row[0] for row in src_cursor.fetchall()]

    return pk_columns


def get_primary_key_index_name(
    database_uri: str,
    table_name: str,
    schema: str = "public",
) -> str | None:
    """Returns the name of the primary key index of a table.

    Parameters
    ----------
    database_uri
        The URI of the database to connect to.
    table_name
        The name of the table to get the primary key index from.
    schema
        The schema of the table.

    Returns
    -------
    The name of the primary key index, or None if the table has no primary key.

    """

    with psycopg.connect(database_uri) as src_conn, src_conn.cursor() as src_cursor:
        src_cursor.execute(
            """
                SELECT ci.relname
                FROM pg_index i
                JOIN pg_class ci ON ci.oid = i.indexrelid
                WHERE i.indrelid = %s::regclass
                  AND i.indisprimary
                """,
            (f"{schema}.{table_name}",),
        )
        pk_index_row = src_cursor.fetchone()
        pk_index_name: str | None = pk_index_row[0] if pk_index_row else None

    return pk_index_name


class IndexInfo(NamedTuple):
    """A named tuple containing information about an index."""

    name: str
    definition: str
    is_unique: bool
    using_clause: str


def get_index_info(
    database_uri: str,
    table_name: str,
    schema: str = "public",
) -> list[IndexInfo]:
    """Returns information about the indices of a table.

    Parameters
    ----------
    database_uri
        The URI of the database to connect to.
    table_name
        The name of the table to get the index information from.
    schema
        The schema of the table.

    Returns
    -------
    A list of `.IndexInfo` named tuples, each containing information about an index.

    """

    with psycopg.connect(database_uri) as src_conn, src_conn.cursor() as src_cursor:
        src_cursor.execute(
            "SELECT indexdef FROM pg_indexes WHERE schemaname = %s AND tablename = %s",
            (schema, table_name),
        )
        index_defs = [row[0] for row in src_cursor.fetchall()]

    index_info: list[IndexInfo] = []

    for index_def in index_defs:
        match = re.match(
            r"CREATE (UNIQUE )?INDEX (\S+) ON \S+ (USING .+)$",
            index_def,
        )
        if not match:
            continue

        is_unique, index_name, using_clause = match.groups()

        index_info.append(
            IndexInfo(
                name=index_name,
                definition=index_def,
                is_unique=bool(is_unique),
                using_clause=using_clause,
            )
        )

    return index_info


def copy_table(
    source_uri: str,
    target_uri: str,
    table_name: str,
    schema: str = "public",
    target_table_name: str | None = None,
    target_schema: str | None = None,
    create_indices: bool = True,
    create_primary_key: bool = True,
    if_exists: Literal["fail", "replace", "append"] = "fail",
    quiet: bool = False,
) -> None:
    """Copies a table, and optionally its indices, from one database to another.

    Both databases can be on different servers. Data is streamed directly between
    the two connections using ``COPY``, without materialising the full table in
    memory. Rows are copied in batches so that very large tables can be copied
    without exhausting memory on either side of the connection.

    Parameters
    ----------
    source_uri
        The URI of the database to copy the table from.
    target_uri
        The URI of the database to copy the table to.
    table_name
        The name of the table to copy.
    schema
        The schema of the source table.
    target_table_name
        The name of the table to create in the target database. Defaults to
        ``table_name``.
    target_schema
        The schema of the target table. Defaults to ``schema``.
    create_indices
        Whether to recreate the indices defined on the source table in the
        target table.
    create_primary_key
        Whether to recreate the primary key constraint of the source table, if
        any, on the target table.
    if_exists
        What to do if the target table already exists. One of ``"fail"``,
        ``"replace"``, or ``"append"``. If ``"append"``, the table must already
        have a compatible schema and no new table or indices are created.
    quiet
        Whether to suppress logging output while copying the table.

    """

    cl = Console(quiet=quiet)

    dt = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    cl.print(rf"[blue]\[i][/] Starting copy process on {dt} UTC.")

    if if_exists not in ("fail", "replace", "append"):
        raise ValueError("if_exists must be one of 'fail', 'replace', or 'append'.")

    target_schema = target_schema or schema
    target_table_name = target_table_name or table_name

    src_qual = f"{schema}.{table_name}"
    tgt_qual = f"{target_schema}.{target_table_name}"

    if not table_exists(source_uri, table_name, schema):
        raise ValueError(f"Table {src_qual} does not exist in the source database.")

    # Get information about columns and pks.
    columns = get_table_columns(source_uri, table_name, schema)
    pk_columns = get_primary_key_columns(source_uri, table_name, schema)
    pk_index_name = get_primary_key_index_name(source_uri, table_name, schema)

    if not columns:
        raise ValueError(f"Table {src_qual} does not exist or has no columns.")

    target_exists = table_exists(target_uri, target_table_name, target_schema)

    source_table_ident = sql.Identifier(schema, table_name)
    target_table_ident = sql.Identifier(target_schema, target_table_name)

    # Create the table if needed.
    with psycopg.connect(target_uri) as tgt_conn, tgt_conn.cursor() as tgt_cursor:
        if target_exists:
            if if_exists == "fail":
                raise ValueError(
                    f"Table {tgt_qual} already exists in the target database."
                )
            elif if_exists == "replace":
                tgt_cursor.execute(sql.SQL("DROP TABLE {}").format(target_table_ident))
                target_exists = False
                cl.print(
                    f"[yellow][!][/] Dropped existing table [blue]{tgt_qual}[/] "
                    "in destination database."
                )

        if not target_exists:
            tgt_cursor.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    sql.Identifier(target_schema)
                )
            )

            column_defs = sql.SQL(", ").join(
                sql.SQL("{} {}{}").format(
                    sql.Identifier(col_name),
                    sql.SQL(data_type),  # INJECTION! # type: ignore
                    sql.SQL(" NOT NULL" if not_null else ""),
                )
                for col_name, data_type, not_null in columns
            )

            tgt_cursor.execute(
                sql.SQL("CREATE TABLE {} ({})").format(
                    target_table_ident,
                    column_defs,
                )
            )

            cl.print(f"[green][✔][/] Created table [blue]{tgt_qual}[/].")

        tgt_conn.commit()

    # Copy the data from one database to the other. We do not use batches as it
    # seems this way of copying things does so in a memory-friendly way.
    with psycopg.connect(source_uri) as src_conn, src_conn.cursor() as src_cursor:
        with psycopg.connect(target_uri) as tgt_conn, tgt_conn.cursor() as tgt_cursor:
            # Lock the table in share mode to prevent concurrent modifications
            # during the copy.
            src_cursor.execute(
                sql.SQL("LOCK TABLE {} IN SHARE MODE").format(source_table_ident)
            )

            copy_in = sql.SQL("COPY {} FROM STDIN (FORMAT BINARY)").format(
                target_table_ident
            )
            copy_out = sql.SQL("COPY {} TO STDOUT (FORMAT BINARY)").format(
                source_table_ident
            )

            with (
                Timer() as timer,
                cl.status(f"Copying [blue]{src_qual}[/] to [blue]{tgt_qual}[/] ..."),
                src_cursor.copy(copy_out) as src_copy,
                tgt_cursor.copy(copy_in) as tgt_copy,
            ):
                for data in src_copy:
                    tgt_copy.write(data)

            tgt_conn.commit()

    cl.print(
        f"[green][✔][/] Copied [blue]{src_qual}[/] to [blue]{tgt_qual}[/] "
        f"in {timer.elapsed:,.1f} seconds."
    )

    # Create the primary key in the target table.
    if create_primary_key and pk_columns and if_exists != "append":
        with (
            Timer() as timer,
            cl.status(f"Creating primary key on [blue]{tgt_qual}[/] ..."),
            psycopg.connect(target_uri) as tgt_conn,
            tgt_conn.cursor() as tgt_cursor,
        ):
            tgt_cursor.execute(
                sql.SQL("ALTER TABLE {} ADD PRIMARY KEY ({})").format(
                    target_table_ident,
                    sql.SQL(", ").join(sql.Identifier(c) for c in pk_columns),
                )
            )
            tgt_conn.commit()

            cl.print(
                f"[green][✔][/] Created primary key on [blue]{tgt_qual}[/] "
                f"in {timer.elapsed:,.1f} seconds."
            )

    # Create the indices in the target table.
    if create_indices and if_exists != "append":
        index_info = get_index_info(source_uri, table_name, schema)

        renamed = (schema, table_name) != (target_schema, target_table_name)

        with (
            Timer() as timer,
            cl.status(f"Creating indices on [blue]{tgt_qual}[/] ..."),
            psycopg.connect(target_uri) as tgt_conn,
            tgt_conn.cursor() as tgt_cursor,
        ):
            for index in index_info:
                # The primary key index was already created above.
                if create_primary_key and pk_columns and index.name == pk_index_name:
                    continue

                if renamed:
                    index_name = f"{target_table_name}_{index.name}"
                else:
                    index_name = index.name

                new_index_def = sql.SQL("CREATE {}INDEX {} ON {} {}").format(
                    sql.SQL("UNIQUE " if index.is_unique else ""),
                    sql.Identifier(index_name),
                    target_table_ident,
                    sql.SQL(index.using_clause),  # INJECTION! # type: ignore
                )

                # We allow failures while creating indices.
                with cl.status(f"Creating index [blue]{index_name}[/] ..."):
                    try:
                        tgt_cursor.execute(new_index_def)
                    except Exception:
                        cl.print(
                            "[yellow][!][/] Failed to create index "
                            f"[blue]{index_name}[/]."
                        )
                        tgt_conn.rollback()

            tgt_conn.commit()

        cl.print(
            f"[green][✔][/] Created indices on [blue]{tgt_qual}[/] "
            f"in {timer.elapsed:,.1f} seconds."
        )


def function_to_composed(func: str) -> sql.Composed:
    """Converts a function call string to a psycopg ``Composed`` object."""

    match = re.match(r"(\w+)\((.*)\)", func)
    if not match:
        raise ValueError(f"Invalid function call: {func}")

    func_name, args_str = match.groups()
    args = [arg.strip() for arg in args_str.split(",") if arg.strip()]

    return sql.Composed(
        [
            sql.Identifier(func_name),
            sql.SQL("("),
            sql.SQL(", ").join(sql.Identifier(arg) for arg in args),
            sql.SQL(")"),
        ]
    )


def get_row_count(
    connection_uri: str,
    table_name: str,
    schema: str = "public",
    approximate=True,
):
    """Returns the row count for a table.

    Parameters
    ----------
    connection_uri
        The database connection URI.
    table
        The table name.
    schema
        The schema in which the table lives.
    approximate
        If True, returns the approximate row count from the ``pg_class``
        table (much faster). Otherwise calculates the exact count.

    """

    if not table_exists(connection_uri, table_name, schema):
        raise ValueError(f"Table {schema}.{table_name} does not exist.")

    with psycopg.connect(connection_uri) as conn, conn.cursor() as cursor:
        if approximate:
            cursor.execute(
                "SELECT reltuples AS approximate_row_count FROM pg_class "
                "JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace "
                "WHERE pg_namespace.nspname = %s AND relname = %s;",
                (
                    schema,
                    table_name,
                ),
            )
        else:
            cursor.execute(
                sql.SQL("SELECT COUNT(*) FROM {};").format(
                    sql.Identifier(schema, table_name)
                )
            )
        count = cursor.fetchall()

    return int(count[0][0])


def download_table_from_db(
    database_uri: str,
    table_name: str,
    schema: str = "public",
    output_path: pathlib.Path | str = "./",
    chunk_size: int | None = 10_000_000,
    quiet: bool = False,
    overwrite: bool = False,
    skip_existing: bool = True,
    use_copy: bool = False,
):
    """Downloads a table from the database and writes it to disk as Parquet files.

    This function provides two very different methods for downloading a table from
    a PostgreSQL database to disk as Parquet files. With ``use_copy=False`` the
    function performs a series of SQL SELECT queries, sorting by the primary key
    and paginating the results to produces files with as many as ``chunk_size`` rows
    each. With ``use_copy=True``, the function uses the PostgreSQL COPY command and the
    results are piped to the ``split`` command, which divides the output into multiple
    CSV files of at most ``chunk_size`` rows each. The intermediate CSV files are then
    converted to Parquet files and the CSV files are deleted.

    ``use_copy=True`` should be preferred for large tables as it is significantly
    faster than the standard SQL SELECT method and has very low memory overhead.
    The main downside is that it requires significant disk space for the intermediate
    CSV files, which are not compressed. For small-ish tables the different may be
    negligible, and using the standard SQL SELECT method might be more convenient.

    Parameters
    ----------
    database_uri
        The URI of the database to connect to.
    table_name
        The name of the table to download.
    schema
        The schema of the table.
    output_path
        the path where the Parquet files will be saved. Each chunk is saved as
        ``<output_path>/<table_name>_<NNNN>.parquet``.
    chunk_size
        The number of rows to read at a time. Each chunk is saved to a separate
        Parquet file.
    quiet
        If True, suppresses progress messages. Otherwise, prints progress messages
        to the console.
    use_copy
        If True, uses the PostgreSQL COPY command for faster downloads. Otherwise,
        falls back to standard SQL SELECT queries.
    overwrite
        If True, overwrites existing Parquet files. Otherwise, raises an error if
        the output files already exist. Not supported for ``use_copy=True``.
    skip_existing
        If True, skips downloading chunks that already exist as Parquet files.
        This is useful for resuming interrupted downloads without re-downloading
        Ignored if ``overwrite`` is True or ``use_copy`` is True.

    """

    if not table_exists(database_uri, table_name, schema):
        raise ValueError(
            f"Table {schema}.{table_name} does not exist or the provided "
            "database_uri is invalid."
        )

    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    cl = Console(quiet=quiet)

    dt = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    cl.print(rf"[blue]\[i][/] Starting download process on {dt} UTC.")

    if use_copy:
        _download_table_using_copy(
            console=cl,
            database_uri=database_uri,
            table_name=table_name,
            schema=schema,
            output_path=output_path,
            chunk_size=chunk_size,
        )
        return

    with (
        Timer() as timer,
        cl.status(f"Downloading table [blue]{schema}.{table_name}[/] ..."),
    ):
        if chunk_size is None:
            output_file = output_path / f"{table_name}_0001.parquet"

            if not overwrite and skip_existing and output_file.exists():
                cl.print(f"[yellow][!][/] Found existing file {output_file!s}.")
                return

            if not overwrite and output_file.exists():
                raise FileExistsError(f"Output file {output_file} already exists.")

            query = f'SELECT * FROM "{schema}"."{table_name}"'
            df = polars.read_database_uri(query, uri=database_uri, engine="adbc")
            df.write_parquet(output_path / f"{table_name}.parquet")

        else:
            n_rows = get_row_count(
                database_uri,
                table_name,
                schema,
                approximate=True,
            )

            n_chunks = n_rows // chunk_size + 1

            nn = 1
            offset = 0

            pk = get_primary_key_columns(database_uri, table_name, schema)
            if not pk:
                raise ValueError(
                    f"Table {schema}.{table_name} has no primary key. "
                    "A primary key is required to download the table in chunks."
                )
            elif len(pk) > 1:
                raise ValueError(
                    f"Table {schema}.{table_name} has a composite primary key. "
                    "A single-column primary key is required to download the table "
                    "in chunks."
                )

            with cl.status(f"Downloading chunk {nn} of {n_chunks} ...") as status:
                while True:
                    status.update(f"Downloading chunk {nn} of {n_chunks} ...")

                    output_file = output_path / f"{table_name}_{nn:04d}.parquet"

                    if not overwrite and skip_existing and output_file.exists():
                        cl.print(f"[yellow][!][/] Found existing file {output_file!s}.")

                        lf = polars.scan_parquet(output_file)
                        n_rows: int = lf.select(polars.len()).collect().item()

                        nn += 1
                        offset += n_rows
                        continue

                    if not overwrite and output_file.exists():
                        raise FileExistsError(f"File {output_file} already exists.")

                    query = (
                        f'SELECT * FROM "{schema}"."{table_name}" '
                        f"ORDER BY {pk[0]} LIMIT {chunk_size} OFFSET {offset}"
                    )
                    df = polars.read_database_uri(
                        query,
                        uri=database_uri,
                        engine="adbc",
                    )

                    df.write_parquet(output_file)

                    if df.is_empty() or df.height < chunk_size:
                        break

                    offset += chunk_size
                    nn += 1

    cl.print(
        f"[green][✔][/] Downloaded table [blue]{schema}.{table_name}[/] "
        f"in {timer.elapsed:,.1f} seconds."
    )

    cl.print(rf"[blue]\[i][/] Table saved to [blue]{output_path.absolute()}[/].")

    return


def _download_table_using_copy(
    console: Console,
    database_uri: str,
    table_name: str,
    schema: str,
    output_path: pathlib.Path,
    chunk_size: int | None,
) -> None:
    """Download a table using the PostgreSQL COPY command.

    This method should be preferred for very large tables where the combination
    of SELECT and OFFSET is inefficient. Instead, we stream the entire table using
    COPY and use the ``split`` command to divide the data into manageable chunks.
    This produces intermediate CSV files which we then convert to Parquet.

    """

    df_schema = get_table_column_types(
        database_uri=database_uri,
        table_name=table_name,
        schema=schema,
    )
    console.print("[green][✔][/] Retrieved table schema.")

    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = list(output_path.glob(f"{table_name}_*.csv"))
    parquet_files = list(output_path.glob(f"{table_name}_*.parquet"))
    if csv_files or parquet_files:
        console.print(
            f"[red][✖][/] CSV or Parquet files found in {output_path.absolute()}. "
            "Delete them or provide a clean path."
        )
        return

    with (
        Timer() as timer,
        console.status(f"COPY-ing table {schema}.{table_name} to CSV files ..."),
    ):
        if chunk_size is None:
            # Simpler case. We just use psycopg's COPY command
            # to download the entire table to a single CSV file.
            with (
                open(output_path / f"{table_name}.csv", "wb") as file_,
                psycopg.connect(database_uri) as conn,
                conn.cursor() as cursor,
                cursor.copy(
                    sql.SQL("COPY {} TO STDOUT WITH (FORMAT CSV, HEADER FALSE)").format(
                        sql.Identifier(schema, table_name)
                    )
                ) as copy,
            ):
                for data in copy:
                    file_.write(data)

        else:
            # We pipe the results of the COPY command into the split command
            # to create chunked CSV files.
            split_command = (
                "split --numeric-suffixes=1 -a 4 "
                f"-l {chunk_size} --additional-suffix=.csv - "
                f"{output_path / f'{table_name}_'}"
            )

            split_process = subprocess.Popen(
                split_command,
                shell=True,
                stdin=subprocess.PIPE,
            )
            assert split_process.stdin is not None

            copy_query = sql.SQL(
                "COPY (SELECT {} from {}) TO STDOUT WITH (FORMAT CSV, HEADER FALSE);"
            ).format(
                sql.SQL(", ").join([sql.Identifier(col) for col in df_schema.keys()]),
                sql.Identifier(schema, table_name),
            )

            with psycopg.connect(database_uri) as conn, conn.cursor() as cursor:
                with cursor.copy(copy_query) as copy:
                    for data in copy:
                        split_process.stdin.write(data)

            split_process.stdin.close()
            split_process.wait()

    console.print(
        f"[green][✔][/] Created CSV files for table [blue]{schema}.{table_name}[/] "
        f"in {timer.elapsed:,.1f} seconds."
    )

    # Now we convert each CSV file to a Parquet file.
    with Timer() as timer, console.status("Converting CSV files to Parquet..."):
        for csv_file in output_path.glob(f"{table_name}_*.csv"):
            parquet_file = csv_file.with_suffix(".parquet")

            (
                polars.scan_csv(csv_file, schema=df_schema)
                .cast(df_schema)  # type: ignore
                .sink_parquet(parquet_file, engine="streaming")
            )

    console.print(
        f"[green][✔][/] Converted CSV files to Parquet in {timer.elapsed:,.1f} seconds."
    )

    # Delete intermediate CSV files.
    for csv_file in output_path.glob(f"{table_name}_*.csv"):
        csv_file.unlink()

    console.print("[green][✔][/] Deleted intermediate CSV files.")


PG_TO_POLARS_TYPES: dict[str, polars.DataType | type[polars.DataType]] = {
    "smallint": polars.Int16,
    "integer": polars.Int32,
    "bigint": polars.Int64,
    "smallserial": polars.Int16,
    "serial": polars.Int32,
    "bigserial": polars.Int64,
    "real": polars.Float32,
    "double precision": polars.Float64,
    "numeric": polars.Float64,
    "decimal": polars.Float64,
    "boolean": polars.Boolean,
    "character varying": polars.Utf8,
    "character": polars.Utf8,
    "varchar": polars.Utf8,
    "text": polars.Utf8,
    "citext": polars.Utf8,
    "uuid": polars.Utf8,
    "json": polars.Utf8,
    "jsonb": polars.Utf8,
    "date": polars.Date,
    "time": polars.Time,
    "time without time zone": polars.Time,
    "time with time zone": polars.Time,
    "timestamp": polars.Datetime,
    "timestamp without time zone": polars.Datetime,
    "timestamp with time zone": polars.Datetime,
    "interval": polars.Duration,
    "bytea": polars.Binary,
}


def get_table_column_types(
    database_uri: str,
    table_name: str,
    schema: str = "public",
) -> dict[str, polars.DataType | type[polars.DataType]]:
    """Return a dictionary of column name to polars type for a database table.

    The column types are read from the ``information_schema.columns`` metadata
    table. Any PostgreSQL type that does not have a known mapping to a polars
    type is mapped to ``polars.Utf8``.

    Parameters
    ----------
    database_uri
        The URI of the database to connect to.
    table_name
        The name of the table to introspect.
    schema
        The schema in which the table is located.

    Returns
    -------
    A mapping of column name to the corresponding polars data type.

    """

    with psycopg.connect(database_uri) as conn, conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
            AND table_schema = %s
            ORDER BY ordinal_position;
            """,
            (table_name, schema),
        )

        rows = cursor.fetchall()

    if not rows:
        raise ValueError(f"Table {schema}.{table_name} not found or has no columns.")

    column_types: dict[str, polars.DataType | type[polars.DataType]] = {}

    for column_name, data_type in rows:
        polars_type = PG_TO_POLARS_TYPES.get(data_type.lower(), polars.Utf8)
        column_types[column_name] = polars_type

    return column_types
