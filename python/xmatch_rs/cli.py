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


@xmatch_rs_cli.group(
    "hats",
    cls=NaturalOrderGroup,
    context_settings={"obj": {}},
    no_args_is_help=True,
    help="Routines to generate and handle HATS datasets.",
)
def hats_cli():
    """Routines to generate and handle HATS datasets."""


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
@cloup.option(
    "--schema",
    "-s",
    type=str,
    default=None,
    help=(
        "Path to a catalogue schema YAML file, or the name of an internally "
        "defined schema. If omitted, the output path must include a _convert.yaml "
        "file with the schema definition."
    ),
)
@cloup.option(
    "--format",
    type=cloup.Choice(["csv", "parquet"]),
    default=None,
    help="Input format. If omitted, inferred from the input files.",
)
@cloup.option(
    "--initial-order",
    type=int,
    default=2,
    show_default=True,
    help="Initial HEALPix order used for partitioning.",
)
@cloup.option(
    "--reference-order",
    type=int,
    default=29,
    show_default=True,
    help="Reference HEALPix order for the _healpix column.",
)
@cloup.option(
    "--max-order",
    type=int,
    default=None,
    help="Maximum HEALPix order for adaptive tiling.",
)
@cloup.option(
    "--max-rows-per-pixel",
    type=int,
    default=1_000_000,
    show_default=True,
    help="Maximum number of rows per pixel.",
)
@cloup.option(
    "--max-kilobytes-per-pixel",
    type=int,
    default=None,
    help="Maximum number of kilobytes per pixel.",
)
@cloup.option(
    "--n-workers",
    "-n",
    type=int,
    default=None,
    help=(
        "Number of worker processes to use. If not provided, automatically "
        "determined based on the number of CPU cores and the task being run."
    ),
)
@cloup.option(
    "--dynamic-planning/--no-dynamic-planning",
    default=True,
    help="Enable/disable dynamic conversion planning.",
)
@cloup.option(
    "--show-progress/--no-show-progress",
    default=True,
    help="Enable/disable progress tracking.",
)
@cloup.option_group(
    "Logging options",
    "Options to control logging output.",
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
def convert(
    data_path: tuple[str, ...],
    collection_path: pathlib.Path,
    schema: pathlib.Path | None,
    format: Literal["csv", "parquet"] | None,
    initial_order: int,
    reference_order: int,
    max_order: int | None,
    max_rows_per_pixel: int | None,
    max_kilobytes_per_pixel: int | None,
    n_workers: int | None,
    dynamic_planning: bool,
    show_progress: bool,
    quiet: bool,
    verbose: bool,
):
    """Convert a dataset to HATS format."""

    from xmatch_rs import log
    from xmatch_rs.hats import HATS

    if verbose:
        log.set_level(10)
    elif quiet:
        log.set_level(40)
    else:
        log.set_level(20)

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
        n_workers=n_workers,
        dynamic_planning=dynamic_planning,
        show_progress=show_progress if not quiet else False,
        log=log,
    )
