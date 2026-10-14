#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-05-16
# @Filename: cli.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import Literal

import click
import cloup


class NaturalOrderGroup(cloup.Group):
    """A group that lists commands in order of definition."""

    def list_commands(self, ctx):
        return self.commands.keys()


def version_callback(ctx: click.Context, _param: click.Parameter, value: bool):
    """Prints version and exits early."""

    if not value or ctx.resilient_parsing:
        return

    from xmatch_rs import __version__

    click.echo(f"xmatch-rs {__version__}")
    ctx.exit()


@cloup.group(
    cls=NaturalOrderGroup,
    context_settings={"obj": {}},
    no_args_is_help=True,
    help="CLI for xmatch-rs.",
)
@cloup.option(
    "--version",
    "-V",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=version_callback,
    help="Show the version and exit.",
)
def xmatch_rs_cli():
    """CLI for xmatch-rs."""

    pass


@xmatch_rs_cli.group(
    "hats",
    cls=NaturalOrderGroup,
    context_settings={"obj": {}},
    no_args_is_help=True,
    help="Routines to generate and handle HATS datasets.",
)
def hats_cli():
    """Routines to generate and handle HATS datasets."""

    pass


@hats_cli.command("convert")
@cloup.argument(
    "data_paths",
    nargs=-1,
    required=True,
)
@cloup.argument(
    "collection_path",
    type=cloup.Path(exists=False, path_type=pathlib.Path),
)
@cloup.option(
    "--schema",
    type=cloup.Path(exists=True, path_type=pathlib.Path, dir_okay=False),
    help="Path to the catalogue schema YAML file.",
)
@cloup.option(
    "--format",
    type=cloup.Choice(["csv", "parquet"]),
    default=None,
    help="Input format. If omitted, inferred from the input files.",
)
@cloup.option(
    "--reference-order",
    type=int,
    default=29,
    show_default=True,
    help="Reference HEALPix order for the _healpix column.",
)
@cloup.option_group(
    "Partitioning options",
    "Options to control how the dataset is partitioned into HEALPix pixels.",
    cloup.option(
        "--initial-order",
        type=int,
        default=2,
        show_default=True,
        help="Initial HEALPix order used for partitioning.",
    ),
    cloup.option(
        "--max-order",
        type=int,
        default=None,
        help="Maximum HEALPix order for adaptive tiling.",
    ),
    cloup.option(
        "--max-rows-per-pixel",
        type=int,
        default=1_000_000,
        show_default=True,
        help="Maximum number of rows per pixel.",
    ),
    cloup.option(
        "--max-kilobytes-per-pixel",
        type=int,
        default=None,
        help="Maximum number of kilobytes per pixel.",
    ),
    cloup.option(
        "--n-workers",
        "-w",
        type=int,
        default=None,
        help=(
            "Number of worker processes to use. If not provided, automatically "
            "determined based on the number of CPU cores and the task being run."
        ),
    ),
)
@cloup.option_group(
    "Rechunk options",
    "Options to control object catalogue rechunking.",
    cloup.option(
        "--rechunk/--no-rechunk",
        default=True,
        help="Whether to rechunk the object catalogue after partitioning.",
    ),
    cloup.option(
        "--max-rows-per-file", type=int, help="Maximum number of rows per file."
    ),
    cloup.option(
        "--max-kilobytes-per-file",
        type=int,
        help="Maximum number of kilobytes per file.",
    ),
)
@cloup.option_group(
    "Primary key index",
    "Options to control the creation of a primary key index.",
    cloup.option(
        "--create-primary-key-index/--no-create-primary-key-index",
        default=True,
        help="Whether to create an index on the catalogue primary key column.",
    ),
    cloup.option(
        "--primary-key-column",
        type=str,
        help="Column that contains the primary key. If not provided, "
        "the primary key column will be inferred from the schema.",
    ),
    cloup.option(
        "--extra-index-columns",
        type=str,
        default="",
        help=("Comma-separated list of additional columns to include in the index."),
    ),
    cloup.option(
        "--check-index-duplicates/--no-check-index-duplicates",
        default=False,
        help="Whether to check for duplicate entries in the primary key index. "
        "This may lead to high memory use for large datasets.",
    ),
    cloup.option(
        "--sort-index",
        is_flag=True,
        default=False,
        help="Whether to sort the primary key index by the primary key column. "
        "Sorting can cause OOM errors for large datasets, so it is disabled by "
        "default.",
    ),
)
@cloup.option(
    "--dynamic-planning/--no-dynamic-planning",
    default=True,
    help="Enable/disable dynamic conversion planning.",
)
@cloup.option_group(
    "Logging options",
    "Options to control logging output.",
    cloup.option(
        "--show-progress/--no-show-progress",
        default=True,
        help="Enable/disable progress tracking.",
    ),
    cloup.option(
        "--quiet",
        "-q",
        is_flag=True,
        default=False,
        help="Suppress logging output.",
    ),
    cloup.option(
        "--verbose",
        "-v",
        is_flag=True,
        default=False,
        help="Enable verbose logging output.",
    ),
    constraint=cloup.constraints.mutually_exclusive,
)
@cloup.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Allow overwriting existing output directory.",
)
def convert(
    data_paths: tuple[str, ...],
    collection_path: pathlib.Path,
    schema: pathlib.Path | None,
    format: Literal["csv", "parquet"] | None,
    initial_order: int,
    reference_order: int,
    max_order: int | None,
    max_rows_per_pixel: int | None,
    max_kilobytes_per_pixel: int | None,
    rechunk: bool,
    max_rows_per_file: int | None,
    max_kilobytes_per_file: int | None,
    create_primary_key_index: bool,
    primary_key_column: str | None,
    extra_index_columns: str,
    check_index_duplicates: bool,
    sort_index: bool,
    n_workers: int | None,
    dynamic_planning: bool,
    show_progress: bool,
    quiet: bool,
    verbose: bool,
    overwrite: bool,
):
    """Convert a dataset to HATS format."""

    from xmatch_rs.hats import HATS
    from xmatch_rs.tools import get_xmatch_rs_logger

    log = get_xmatch_rs_logger("xmatch_rs.cli.convert")

    if verbose:
        log.set_level(10)
    elif quiet:
        log.set_level(40)
    else:
        log.set_level(20)

    extra_index_columns_list = (
        extra_index_columns.strip().split(",")
        if extra_index_columns.strip() != ""
        else []
    )

    HATS.convert_to_hats(
        data_path=list(data_paths),
        collection_path=collection_path,
        schema=schema,
        format=format,
        initial_order=initial_order,
        reference_order=reference_order,
        max_order=max_order,
        max_rows_per_pixel=max_rows_per_pixel,
        max_kilobytes_per_pixel=max_kilobytes_per_pixel,
        rechunk=rechunk,
        max_rows_per_file=max_rows_per_file,
        max_kilobytes_per_file=max_kilobytes_per_file,
        n_workers=n_workers,
        dynamic_planning=dynamic_planning,
        show_progress=show_progress if not quiet else False,
        log=log,
        overwrite=overwrite,
        create_primary_key_index=create_primary_key_index,
        primary_key_column=primary_key_column,
        extra_index_columns=extra_index_columns_list,
        check_index_duplicates=check_index_duplicates,
        sort_index=sort_index,
    )


@hats_cli.command("create-index")
@cloup.argument(
    "collection_path",
    type=cloup.Path(exists=True, path_type=pathlib.Path, file_okay=False),
)
@cloup.argument("index_column", type=str)
@cloup.option(
    "--index-name",
    "-n",
    type=str,
    help="Name of the index to create. Defaults to 'index_<INDEX_COLUMN>'.",
)
@cloup.option(
    "--extra-columns",
    "-e",
    type=str,
    default="",
    help="Additional columns to include in the index, separated by commas. "
    "These columns will be included in the index file.",
)
@cloup.option(
    "--mode",
    "-m",
    type=cloup.Choice(["sequential", "full"]),
    default="sequential",
    show_default=True,
    help="Mode to use for creating the index.",
)
@cloup.option(
    "--check-duplicates/--no-check-duplicates",
    default=False,
    help=("Whether to check for duplicate entries in the index."),
)
@cloup.option(
    "--sort",
    is_flag=True,
    default=False,
    help="Whether to sort the index by the index column. Sorting can cause OOM errors "
    "for large datasets, so it is disabled by default.",
)
@cloup.option(
    "--max-rows-per-file",
    type=int,
    help="The maximum number of rows per file when chunking the index.",
)
@cloup.option(
    "--max-kilobytes-per-file",
    type=int,
    help="The maximum number of kilobytes per file when chunking the index.",
)
@cloup.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Allow overwriting existing index dataset.",
)
def create_index(
    collection_path: pathlib.Path,
    index_column: str,
    index_name: str | None,
    extra_columns: str,
    mode: Literal["sequential", "full"],
    check_duplicates: bool = False,
    sort: bool = False,
    max_rows_per_file: int | None = None,
    max_kilobytes_per_file: int | None = None,
    overwrite: bool = False,
):
    """Creates an index dataset from the object catalogue."""

    from xmatch_rs.hats import HATS

    index_column = index_column.lower()
    index_name = index_name or f"index_{index_column.lower()}"

    hats = HATS(collection_path)

    hats.create_index(
        index_name=index_name,
        index_column=index_column,
        extra_columns=extra_columns.strip().split(","),
        check_duplicates=check_duplicates,
        max_rows_per_file=max_rows_per_file,
        max_kilobytes_per_file=max_kilobytes_per_file,
        mode=mode,
        overwrite=overwrite,
        sort=sort,
    )


@hats_cli.command("create-margin")
@cloup.argument(
    "collection_path",
    type=cloup.Path(exists=True, path_type=pathlib.Path, file_okay=False),
)
@cloup.argument("buffer", type=int)
@cloup.option(
    "--margin-name",
    "-n",
    type=str,
    help="Name of the margin cache to create. Defaults to 'margin_<BUFFER>arcsec'.",
)
@cloup.option(
    "--default-margin",
    is_flag=True,
    default=False,
    help="Sets the margin cache as the default margin in the collection properties.",
)
@cloup.option(
    "--n-workers",
    "-w",
    type=int,
    default=None,
    help=(
        "Number of worker processes to use. If not provided, automatically "
        "determined based on the number of CPU cores."
    ),
)
@cloup.option(
    "--show-progress/--no-show-progress",
    default=True,
    help="Enable/disable progress tracking.",
)
@cloup.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress logging output.",
)
@cloup.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Allow overwriting existing margin cache.",
)
def create_margin(
    collection_path: pathlib.Path,
    buffer: int,
    margin_name: str | None = None,
    default_margin: bool = False,
    n_workers: int | None = None,
    show_progress: bool = True,
    quiet: bool = False,
    overwrite: bool = False,
):
    """Creates a margin cache for a HATS collection."""

    from xmatch_rs.hats import HATS

    margin_name = margin_name or f"margin_{int(buffer)}arcsec"

    hats = HATS(collection_path)

    hats.create_margin_cache(
        margin_name=margin_name,
        buffer=buffer,
        n_workers=n_workers,
        show_progress=show_progress,
        quiet=quiet,
        overwrite=overwrite,
        default_margin=default_margin,
    )


@xmatch_rs_cli.group(
    "schema",
    cls=NaturalOrderGroup,
    context_settings={"obj": {}},
    no_args_is_help=True,
    help="Handle catalogue schemas.",
)
def schema_cli():
    """Handle catalogue schemas."""

    pass


@schema_cli.command("validate")
@cloup.argument(
    "schema_path",
    type=cloup.Path(exists=True, path_type=pathlib.Path, dir_okay=False),
)
@cloup.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed output.",
)
def validate_schema(schema_path: pathlib.Path, verbose: bool = False):
    """Validates a catalogue schema YAML file."""

    from pydantic import ValidationError

    from xmatch_rs import console
    from xmatch_rs.schema import Schema

    try:
        Schema(schema_path)
    except ValidationError as err:
        console.print(f"[red]Schema validation failed for {schema_path!s}[/]:\n{err}")
        raise click.exceptions.Exit(65)  # Data format error.

    if verbose:
        print(f"Schema {schema_path} is valid.")


@schema_cli.command("generate")
@cloup.argument(
    "DATA_PATH",
    type=cloup.Path(exists=True, path_type=pathlib.Path),
    required=False,
)
@cloup.option(
    "--output-path",
    "-o",
    type=cloup.Path(exists=False, path_type=pathlib.Path, dir_okay=False),
    help="Path to save the generated schema YAML file. If not provided, "
    "the schema will be printed to stdout.",
)
@cloup.option(
    "--format",
    type=cloup.Choice(["csv", "parquet", "fits"]),
    default=None,
    help="Data format. If not provided, inferred from the file extension.",
)
@cloup.option(
    "--skip-optional/--no-skip-optional",
    default=False,
    help="Whether to skip optional fields and set them to their default value.",
)
@cloup.option(
    "--csv-has-header",
    is_flag=True,
    default=True,
    help="For CSV files, whether the first row contains column names.",
)
@cloup.option(
    "--csv-delimiter",
    type=str,
    default=",",
    show_default=True,
    help="Delimiter to use for CSV files.",
)
@cloup.option(
    "--csv-comment-prefix",
    type=str,
    default="#",
    show_default=True,
    help="Comment prefix to use for CSV files.",
)
@cloup.option(
    "--csv-null-values",
    type=str,
    multiple=True,
    default=("null",),
    help="A list of strings that should be interpreted as null values for CSV files.",
)
def generate_schema(
    data_path: pathlib.Path | None = None,
    format: Literal["csv", "parquet", "fits"] | None = None,
    output_path: pathlib.Path | None = None,
    skip_optional: bool = False,
    csv_has_header: bool = True,
    csv_delimiter: str = ",",
    csv_comment_prefix: str = "#",
    csv_null_values: tuple[str, ...] = ("null",),
):
    """Generates a catalogue schema YAML file from a sample dataset."""

    import yaml

    from xmatch_rs import console
    from xmatch_rs.schema import generate_schema

    schema_model = generate_schema(
        data_path,
        format=format,
        skip_optional=skip_optional,
        csv_has_header=csv_has_header,
        csv_delimiter=csv_delimiter,
        csv_comment_prefix=csv_comment_prefix,
        csv_null_values=list(csv_null_values),
    )

    if schema_model is None:
        console.print("[red]Failed to generate schema.[/red]")
        raise click.exceptions.Exit(1)

    if output_path:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(
                schema_model.model_dump(mode="json"),
                f,
                sort_keys=False,
                indent=2,
            )

        console.print(f"[green]Schema saved to {output_path!s}[/]")
        return

    print(yaml.dump(schema_model.model_dump(mode="json"), sort_keys=False, indent=2))


@xmatch_rs_cli.group(
    "database",
    cls=NaturalOrderGroup,
    context_settings={"obj": {}},
    no_args_is_help=True,
    help="Handle database operations.",
)
def database_cli():
    """Handle database operations."""

    pass


@database_cli.command("load-hats")
@cloup.argument(
    "DATABASE_URI",
    type=str,
    help="URI of the database to load the HATS collection into.",
)
@cloup.argument(
    "HATS_COLLECTION_PATH",
    type=cloup.Path(
        exists=True,
        path_type=pathlib.Path,
        file_okay=False,
    ),
    help="Path to the HATS collection directory.",
)
@cloup.option(
    "--table",
    type=str,
    required=False,
    help="Name of the table to load the HATS collection into.",
)
@cloup.option(
    "--schema",
    type=str,
    required=False,
    help="Name of the schema to load the HATS collection into.",
)
@cloup.option(
    "--create-table/--no-create-table",
    default=True,
    help="Whether to create the table if it does not exist.",
)
@cloup.option(
    "--create-primary-key/--no-create-primary-key",
    default=True,
    help="Whether to create the primary key on the loaded table.",
)
@cloup.option(
    "--create-indices/--no-create-indices",
    default=True,
    help="Whether to create the indices on the loaded table.",
)
@cloup.option(
    "--n-workers",
    "-w",
    type=int,
    default=None,
    help=("Number of worker processes to use."),
)
@cloup.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress logging output.",
)
def database_load_hats(
    database_uri: str,
    hats_collection_path: pathlib.Path,
    n_workers: int | None = None,
    quiet: bool = False,
    table: str | None = None,
    schema: str | None = None,
    create_table: bool = True,
    create_primary_key: bool = True,
    create_indices: bool = True,
):
    """Load a HATS collection into the database."""

    from xmatch_rs.database.load import write_hats_dataset_to_database
    from xmatch_rs.hats.hats import HATS

    hats = HATS(hats_collection_path)

    write_hats_dataset_to_database(
        hats=hats,
        database_uri=database_uri,
        schema=schema,
        table_name=table,
        create_table=create_table,
        create_primary_key=create_primary_key,
        create_indices=create_indices,
        n_workers=n_workers,
        quiet=quiet,
    )


@database_cli.command("copy-table")
@cloup.argument(
    "SOURCE_DATABASE_URI",
    type=str,
    help="URI of the source database containing the table to copy.",
)
@cloup.argument(
    "TARGET_DATABASE_URI",
    type=str,
    help="URI of the target database to copy the table into.",
)
@cloup.argument(
    "TABLE_NAME",
    type=str,
    help="Name of the table to copy.",
)
@cloup.option(
    "--source-table",
    type=str,
    required=False,
    help="Name of the source table. Defaults to the same as TABLE_NAME.",
)
@cloup.option(
    "--source-schema",
    type=str,
    required=False,
    help="Name of the schema containing the table to copy. "
    "Defaults to the public schema.",
)
@cloup.option(
    "--target-table",
    type=str,
    required=False,
    help="Name of the target table. Defaults to the same as TABLE_NAME.",
)
@cloup.option(
    "--target-schema",
    type=str,
    required=False,
    help="Name of the schema to copy the table into. Defaults to the public schema.",
)
@cloup.option(
    "--create-primary-key/--no-create-primary-key",
    default=True,
    help="Whether to create the primary key on the loaded table.",
)
@cloup.option(
    "--create-indices/--no-create-indices",
    default=True,
    help="Whether to create the indices on the loaded table.",
)
@cloup.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress logging output.",
)
def copy_table(
    source_database_uri: str,
    target_database_uri: str,
    table_name: str,
    source_table: str | None = None,
    source_schema: str | None = None,
    target_table: str | None = None,
    target_schema: str | None = None,
    create_primary_key: bool = True,
    create_indices: bool = True,
    quiet: bool = False,
) -> None:
    """Copies a table from one database to another."""

    from xmatch_rs.database.tools import copy_table as copy_table_fn

    copy_table_fn(
        source_uri=source_database_uri,
        target_uri=target_database_uri,
        table_name=source_table or table_name,
        schema=source_schema or "public",
        target_table_name=target_table or table_name,
        target_schema=target_schema or "public",
        create_primary_key=create_primary_key,
        create_indices=create_indices,
        quiet=quiet,
    )


@database_cli.command("download-table")
@cloup.argument(
    "DATABASE_URI",
    type=str,
    help="URI of the database containing the table to download.",
)
@cloup.argument(
    "TABLE_NAME",
    type=str,
    help="Name of the table to download.",
)
@cloup.argument(
    "OUTPUT_PATH",
    type=cloup.Path(exists=False, path_type=pathlib.Path),
    help="Path to save the downloaded table.",
)
@cloup.option(
    "--schema",
    type=str,
    required=False,
    help="Name of the schema containing the table to download. "
    "Defaults to the public schema.",
)
@cloup.option(
    "--chunk-size",
    type=int,
    required=False,
    default=10_000_000,
    help="Number of rows to download at a time. Defaults to 10_000_000 rows.",
)
@cloup.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress logging output.",
)
@cloup.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Whether to overwrite existing files.",
)
@cloup.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    help="Whether to skip downloading if the output file already exists.",
)
@cloup.option(
    "--use-copy/--no-use-copy",
    default=None,
    help="Whether to use the COPY command for faster downloads. If not specified, "
    "the best mode will be decided automatically based on the table size.",
)
def download_table(
    database_uri: str,
    table_name: str,
    output_path: pathlib.Path,
    schema: str | None = None,
    chunk_size: int = 10_000_000,
    quiet: bool = False,
    overwrite: bool = False,
    skip_existing: bool = True,
    use_copy: bool | None = None,
) -> None:
    """Downloads a table from the specified database."""

    from xmatch_rs.database.tools import download_table_from_db as download_table_fn

    download_table_fn(
        database_uri=database_uri,
        table_name=table_name,
        output_path=output_path,
        schema=schema or "public",
        chunk_size=chunk_size,
        quiet=quiet,
        overwrite=overwrite,
        skip_existing=skip_existing,
        use_copy=use_copy,
    )


@database_cli.command("load-table")
@cloup.argument(
    "DATABASE_URI",
    type=str,
    help="URI of the database containing the table to download.",
)
@cloup.argument(
    "TABLE_NAME",
    type=str,
    help="Name of the table to download.",
)
@cloup.argument(
    "DATA_PATH",
    type=cloup.Path(exists=True, path_type=pathlib.Path),
    nargs=-1,
    help="Path to the directory containing the data to load, in Parquet format.",
)
@cloup.option(
    "--schema",
    type=str,
    required=False,
    help="Name of the schema containing the table to download. "
    "Defaults to the public schema.",
)
@cloup.option(
    "--chunk-size",
    type=int,
    required=False,
    default=1_000_000,
    help="Number of rows to load at a time. Defaults to 1_000_000 rows.",
)
@cloup.option(
    "--create-table/--no-create-table",
    default=True,
    help="Whether to create the table if it does not exist.",
)
@cloup.option(
    "--primary-key",
    type=str,
    help="Name of the primary key column for the table.",
)
@cloup.option(
    "--index",
    type=str,
    multiple=True,
    help="Create an index on this column. Can be specified multiple times.",
)
@cloup.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress logging output.",
)
def load_table(
    database_uri: str,
    table_name: str,
    data_path: tuple[pathlib.Path, ...],
    schema: str | None = None,
    chunk_size: int = 10_000_000,
    create_table: bool = True,
    quiet: bool = False,
    primary_key: str | None = None,
    index: tuple[str, ...] = (),
) -> None:
    """Loads a table into the specified database from a directory of Parquet files."""

    from xmatch_rs.database.load import write_parquet_to_database

    write_parquet_to_database(
        frame=list(data_path),
        database_uri=database_uri,
        table_name=table_name,
        schema=schema or "public",
        chunk_size=chunk_size,
        quiet=quiet,
        create_table=create_table,
        primary_key=primary_key,
        indices=list(index),
    )
