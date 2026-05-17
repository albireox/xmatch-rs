#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-03-18
# @Filename: tools.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import datetime
import glob
import pathlib
import re

import polars

import sdsstools.logger

from xmatch_rs import console


__all__: list[str] = [
    "lazy_frame_row_count",
    "get_polars_dtype_from_string",
    "is_glob_pattern",
    "get_xmatch_rs_logger",
    "parse_path",
    "get_current_time",
]


def lazy_frame_row_count(lf: polars.LazyFrame) -> int:
    """Returns the number of rows in a lazy frame."""

    return lf.select(polars.len()).collect(engine="streaming").item()  # type: ignore


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


def is_glob_pattern(path_string: str | pathlib.Path) -> bool:
    """Checks if a path string contains any glob wildcard characters."""

    path_string = str(path_string)

    # Check for standard glob characters
    if any(char in path_string for char in ["*", "?", "[", "]"]):
        return True

    # Check for recursive glob pattern (**)
    if "**" in path_string:
        # Ensure it's not just a part of a filename that happens to have two asterisks,
        # but that it functions as a recursive glob pattern with path separators.
        # This basic check is sufficient for most cases.
        return True

    return False


def get_xmatch_rs_logger(
    name: str = "xmatch_rs",
    use_rich_handler: bool = True,
    log_level: int = 100,
) -> sdsstools.logger.SDSSLogger:
    """Returns a Polars logger with the specified name."""

    log = sdsstools.logger.get_logger(
        name,
        use_rich_handler=use_rich_handler,
        log_level=log_level,
    )
    log.sh.setLevel(log_level)

    if use_rich_handler:
        log.rich_console = console
        if isinstance(log.sh, sdsstools.logger.RichHandler):
            log.sh.console = console

    return log


def parse_path(
    path: list[str]
    | list[pathlib.Path]
    | list[str | pathlib.Path]
    | str
    | pathlib.Path,
    extension: str | None = None,
    recursive: bool = False,
) -> list[pathlib.Path]:
    """Parses a path, with glob pattern support, and returns a list of files.

    Parameters
    ----------
    path
        The path to parse. Can be a string or a pathlib.Path object. Can include glob
        patterns.
    extension
        If provided, only files with this extension will be returned. Matched as
        a regular expression against the file name.
    recursive
        If True, directories will be searched recursively. Ignored if glob patterns
        are used, as they can already include recursive patterns (e.g., **/*.*).

    Returns
    -------
    list[pathlib.Path]
        A list of Path objects representing the files that match the input path.

    """

    if isinstance(path, (str, pathlib.Path)):
        path = [path]

    path_tmp: list[pathlib.Path | str] = []
    for file in path:
        if is_glob_pattern(file):
            for matched_file in glob.glob(str(file)):
                path_tmp.append(matched_file)
        else:
            path = pathlib.Path(file)
            if path.is_dir():
                files = path.glob("**/*.*") if recursive else path.glob("*.*")
                for matched_file in files:
                    path_tmp.append(matched_file)
            else:
                path_tmp.append(path)

    paths = [pathlib.Path(ff) for ff in path_tmp]

    if extension is not None:
        paths = [ff for ff in paths if re.match(rf".*{extension}$", ff.name)]

    return paths


def get_current_time(format: str = "isot", timezone: str = "UTC") -> str:
    """Returns the current time as a string in the specified format and timezone."""

    if format.lower() != "isot":
        raise NotImplementedError(f"Time format '{format}' is not supported.")

    if timezone.upper() != "UTC":
        raise NotImplementedError(f"Timezone '{timezone}' is not supported.")

    now = datetime.datetime.now(datetime.timezone.utc)

    return now.strftime("%Y-%m-%dT%H:%M:%SZ")
