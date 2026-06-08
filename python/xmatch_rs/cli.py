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
import yaml
from pydantic import ValidationError

from xmatch_rs import console, log


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
    "data_path",
    nargs=-1,
    required=True,
)
@cloup.argument(
    "collection_path",
    type=cloup.Path(exists=False, path_type=pathlib.Path),
)
@cloup.argument(
    "schema",
    type=cloup.Path(exists=True, path_type=pathlib.Path, dir_okay=False),
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
    data_path: tuple[str, ...],
    collection_path: pathlib.Path,
    schema: pathlib.Path,
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
        data_path=list(data_path),
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
    default="null",
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

    from xmatch_rs import generate_schema

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
