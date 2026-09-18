#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-06-06
# @Filename: load.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import functools
import multiprocessing
import pathlib

from typing import TYPE_CHECKING

import polars
import psycopg
from psycopg import sql
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from xmatch_rs import console
from xmatch_rs.database import create_table_sql
from xmatch_rs.database.tools import function_to_composed
from xmatch_rs.hats.tools import get_pixel_path


if TYPE_CHECKING:
    from xmatch_rs import HATS


__all__ = ["write_dataset_to_database", "get_database_uri"]


def get_database_uri(
    dbname: str,
    host: str | None = None,
    port: int | None = None,
    user: str | None = None,
    password: str | None = None,
):
    """Returns the URI to the database."""

    if user is None and password is None:
        auth: str = ""
    elif password is None:
        auth: str = f"{user}@"
    else:
        auth: str = f"{user}:{password}@"

    host_port: str = f"{host or ''}" if port is None else f"{host or ''}:{port}"

    return f"postgresql://{auth}{host_port}/{dbname}"


def write_dataset_to_database(
    hats: HATS,
    dbname: str,
    username: str,
    host: str | None = None,
    show_progress: bool = True,
    n_workers: int | None = None,
    table_name: str | None = None,
    schema: str | None = None,
    load_hats_columns: bool = True,
    create_table: bool = True,
    create_indices: bool = False,
) -> None:
    """Loads a HATS dataset into a database.

    Parameters
    ----------
    hats
        The HATS dataset to load.
    dbname
        The name of the database to load into.
    username
        The username to use to connect to the database. The password will be read from
        ``.pgpass``.
    host
        The host to connect to the database. If None, it will connect to the localhost.
    show_progress
        Whether to show a progress bar while loading the data.
    n_workers
        The number of workers to use for loading the data. If None, uses all
        available CPU cores.
    table_name
        The name of the table to load the data into. This can be used to override
        the table name defined in the HATS schema.
    schema
        The name of the schema to load the data into. This can be used to override
        the schema name defined in the HATS schema.
    load_hats_columns
        Whether to load the HATS-specific columns (``healpix_29``,
        ``hats_norder``, ...).
    create_table
        Whether to create the table defined in the HATS schema if it does not exist.
    create_indices
        Whether to create the indices defined in the ``database.indices`` section
        of the schema after the data has been loaded.

    """

    dataset_path = hats.path / "object"
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"No dataset found at {dataset_path!r}.")

    schema = schema or hats.schema.model.database.schema_
    table_name = table_name or hats.schema.model.database.table_name

    partition_info = hats.read_partition_info()
    pixel_data: list[tuple[int, int]] = []
    for row in partition_info.iter_rows(named=True):
        norder = row["Norder"]
        npix = row["Npix"]

        pixel_data.append((norder, npix))

    db_uri = get_database_uri(dbname=dbname, host=host, user=username)

    try:
        with psycopg.connect(db_uri) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = %s
                          AND table_name = %s
                    )
                    """,
                    (schema, table_name),
                )
                table_exists = cursor.fetchone()
    except psycopg.Error as exc:
        raise ConnectionError(f"Could not connect to database {dbname!r}.") from exc

    if table_exists is None or not table_exists[0]:
        if not create_table:
            raise ValueError(
                f"Table {schema}.{table_name} does not exist in database {dbname!r}."
            )
        else:
            if show_progress:
                console.print(
                    f"Creating table {schema}.{table_name} in database {dbname!r} ..."
                )

            with psycopg.connect(db_uri) as conn:
                with conn.cursor() as cursor:
                    # This is SQL injection of the worst kind but we are generating it
                    # and we trust the code so let's just do it.
                    cursor.execute(create_table_sql(hats))  # INJECTION! # type: ignore

    pbar = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )

    pbar_task: int | None = None
    if show_progress:
        pbar_task = pbar.add_task(
            "Loading dataset into database...",
            total=partition_info.height,
        )

        pbar.start()

    try:
        _write_pixel_to_database_partial = functools.partial(
            _write_pixel_to_database,
            dataset_path=dataset_path,
            db_uri=db_uri,
            schema=schema,
            table_name=table_name,
            load_hats_columns=load_hats_columns,
        )

        if n_workers is None:
            n_workers = multiprocessing.cpu_count()

        with multiprocessing.Pool(n_workers) as pool:
            for _ in pool.imap_unordered(
                _write_pixel_to_database_partial,
                pixel_data,
            ):
                if pbar_task is not None:
                    pbar.update(pbar_task, advance=1)
    except Exception:
        if pbar_task is not None:
            pbar.stop()
        raise

    if pbar_task is not None:
        pbar.stop()

    if create_indices:
        indices = hats.schema.model.database.indices
        if len(indices) > 0:
            if show_progress:
                console.print("Creating indices ...")

            _create_indices(
                db_uri,
                schema=schema,
                table_name=table_name,
                indices=indices,
            )


def _create_indices(
    db_uri: str,
    schema: str,
    table_name: str,
    indices: list[str | list[str]],
) -> None:
    """Creates the indices defined in the schema for a table."""

    with psycopg.connect(db_uri) as conn:
        with conn.cursor() as cursor:
            for index in indices:
                # Depending on the type of index (list of columns, single column, or
                # function), we need to create the appropriate SQL object for psycopg.
                if isinstance(index, list):
                    columns = sql.SQL(", ").join(sql.Identifier(col) for col in index)
                else:
                    if "(" in index:
                        columns = function_to_composed(index)
                    else:
                        columns = sql.Identifier(index)

                # Runs the CREATE INDEX statement.
                cursor.execute(
                    sql.SQL("CREATE INDEX ON {} ({})").format(
                        sql.Identifier(schema, table_name),
                        columns,
                    )
                )


def _write_pixel_to_database(
    pixel_data: tuple[int, int],
    dataset_path: pathlib.Path,
    db_uri: str,
    schema: str,
    table_name: str,
    load_hats_columns: bool,
) -> None:
    """Writes a single pixel to the database. This is used for multiprocessing."""

    norder, npix = pixel_data
    dir_ = npix // 10_000 * 10_000

    pixel_path = get_pixel_path(dataset_path, norder, npix)

    data = polars.read_parquet(pixel_path)

    if load_hats_columns:
        data = data.rename({"_healpix_29": "healpix_29"})
        data = data.with_columns(
            polars.lit(norder, dtype=polars.UInt8).alias("hats_norder"),
            polars.lit(dir_, dtype=polars.UInt32).alias("hats_dir"),
            polars.lit(npix, dtype=polars.UInt64).alias("hats_npix"),
        )
    else:
        data = data.drop(["_healpix_29"])

    data.write_database(
        table_name=f"{schema}.{table_name}",
        connection=db_uri,
        if_table_exists="append",
        engine="adbc",
    )
