#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-06-06
# @Filename: load.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import datetime
import functools
import multiprocessing
import pathlib

from typing import TYPE_CHECKING

import polars
import psycopg
from psycopg import sql
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from sdsstools.utils import Timer

from xmatch_rs.database import create_table_sql
from xmatch_rs.database.tools import function_to_composed, rich_status
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
    database_uri: str,
    show_progress: bool = True,
    n_workers: int | None = None,
    table_name: str | None = None,
    schema: str | None = None,
    load_hats_columns: bool = True,
    create_table: bool = True,
    create_primary_key: bool = True,
    create_indices: bool = True,
    quiet: bool = False,
) -> None:
    """Loads a HATS dataset into a database.

    Parameters
    ----------
    hats
        The HATS dataset to load.
    database_uri
        The URI of the database to load into.
    show_progress
        Whether to show a progress bar while loading the data.
    n_workers
        The number of workers to use for loading the data. If None, uses a single core.
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
    create_primary_key
        Whether to create the primary key defined in the ``primary_key``
        section of the schema after the data has been loaded.
    create_indices
        Whether to create the indices defined in the ``database.indices`` section
        of the schema after the data has been loaded.
    quiet
        Whether to suppress logging output while loading the data.

    """

    cl = Console(quiet=quiet)

    dt = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    cl.print(rf"[blue]\[i][/] Starting DB write process on {dt} UTC.")

    dataset_path = hats.path / "object"
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"No dataset found at {dataset_path!r}.")

    schema = schema or hats.schema.model.database.schema_
    table_name = table_name or hats.schema.model.database.table_name
    tbl_qual = f"{schema}.{table_name}"

    partition_info = hats.read_partition_info()
    pixel_data: list[tuple[int, int]] = []
    for row in partition_info.iter_rows(named=True):
        norder = row["Norder"]
        npix = row["Npix"]

        pixel_data.append((norder, npix))

    try:
        with psycopg.connect(database_uri) as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                )
                """,
                (schema, table_name),
            )
            table_exists = cursor.fetchone()
    except psycopg.Error as exc:
        raise ConnectionError("Could not connect to the database.") from exc

    if table_exists is None or not table_exists[0]:
        if not create_table:
            raise ValueError(
                f"Table {schema}.{table_name} does not exist in the database."
            )
        else:
            with psycopg.connect(database_uri) as conn, conn.cursor() as cursor:
                # This is SQL injection of the worst kind but we are generating it
                # and we trust the code so let's just do it.
                cursor.execute(create_table_sql(hats))  # INJECTION! # type: ignore

            cl.print(
                f"[green][✔][/] Created table [blue]{schema}.{table_name}[/] "
                f"in the database."
            )

    pbar = Progress(
        SpinnerColumn("bouncingBar", style="blue"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=cl if not quiet else Console(),
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
            database_uri=database_uri,
            schema=schema,
            table_name=table_name,
            load_hats_columns=load_hats_columns,
        )

        if n_workers is None:
            n_workers = 1

        with Timer() as timer:
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

    cl.print(
        f"[green][✔][/] Loaded dataset into database [blue]{tbl_qual}[/] "
        f"in {timer.elapsed:,.1f} seconds."
    )

    if create_primary_key:
        primary_key = hats.schema.model.primary_key
        if primary_key:
            with (
                Timer() as timer,
                rich_status(cl, f"Creating PK on [blue]{tbl_qual}[/] ..."),
                psycopg.connect(database_uri) as conn,
                conn.cursor() as cursor,
            ):
                cursor.execute(
                    sql.SQL("ALTER TABLE {} ADD PRIMARY KEY ({})").format(
                        sql.Identifier(schema, table_name),
                        sql.Identifier(primary_key),
                    )
                )

            cl.print(
                f"[green][✔][/] Created PK on [blue]{tbl_qual}[/] "
                f"in {timer.elapsed:,.1f} seconds."
            )

    if create_indices:
        indices = hats.schema.model.database.indices
        if len(indices) > 0:
            with (
                Timer() as timer,
                rich_status(cl, f"Creating indices on [blue]{tbl_qual}[/] ..."),
                psycopg.connect(database_uri) as conn,
                conn.cursor() as cursor,
            ):
                for index in indices:
                    # Depending on the type of index (list of columns, single column, or
                    # function), we need to create the appropriate SQL object.
                    if isinstance(index, list):
                        columns = sql.SQL(", ").join(
                            sql.Identifier(col) for col in index
                        )
                    else:
                        if "(" in index:
                            columns = function_to_composed(index)
                        else:
                            columns = sql.Identifier(index)

                    # Runs the CREATE INDEX statement.
                    with rich_status(cl, f"Creating index [blue]{index}[/] ..."):
                        cursor.execute(
                            sql.SQL("CREATE INDEX ON {} ({})").format(
                                sql.Identifier(schema, table_name),
                                columns,
                            )
                        )

            cl.print(
                f"[green][✔][/] Created indices on [blue]{schema}.{table_name}[/] "
                f"in {timer.elapsed:,.1f} seconds."
            )


def _write_pixel_to_database(
    pixel_data: tuple[int, int],
    dataset_path: pathlib.Path,
    database_uri: str,
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
        connection=database_uri,
        if_table_exists="append",
        engine="adbc",
    )
