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

from typing import Type

import numpy
import polars
import polars._typing

import sdsstools.logger

from xmatch_rs import console


__all__: list[str] = [
    "lazy_frame_row_count",
    "get_polars_dtype",
    "is_glob_pattern",
    "get_xmatch_rs_logger",
    "list_files",
    "get_current_time",
    "sink_to_file",
]


def lazy_frame_row_count(lf: polars.LazyFrame) -> int:
    """Returns the number of rows in a lazy frame."""

    return lf.select(polars.len()).collect(engine="streaming").item()


def get_polars_dtype(
    dtype: polars.DataType | str | numpy.dtype,
    inner: Type[polars.DataType] | None = None,
    shape: tuple[int, ...] | None = None,
) -> tuple[
    Type[polars.DataType],
    Type[polars.DataType] | Type[polars._typing.DataTypeClass] | None,
    tuple[int, ...] | None,
]:
    """Retrieves a Polars DataType object.

    Parameters
    ----------
        dtype
        The name of the Polars data type as a string (e.g., "Int64", "Utf8", "Date"),
        a numpy dtype string (e.g., "int64", "float32", "u4"), or a numpy dtype object.

    Returns
    -------
    Type[polars.DataType]
        A Polars DataType class.

    """

    if isinstance(dtype, polars.DataType):
        if isinstance(dtype, polars.Array):
            return (polars.Array, type(inner or dtype.inner), shape or dtype.shape)
        elif isinstance(dtype, polars.List):
            return (polars.List, type(inner or dtype.inner), None)
        else:
            return (type(dtype), inner, shape)

    if isinstance(dtype, str):
        try:
            return (getattr(polars, dtype), inner, shape)
        except AttributeError:
            pass

    if isinstance(dtype, numpy.dtype):
        dtype = dtype.name

    dtype_str = str(dtype).lower()

    if dtype_str in ["int8", "i1"]:
        return (polars.Int8, None, None)
    elif dtype_str in ["int16", "i2"]:
        return (polars.Int16, None, None)
    elif dtype_str in ["int32", "i4"]:
        return (polars.Int32, None, None)
    elif dtype_str in ["int64", "i8"]:
        return (polars.Int64, None, None)
    elif dtype_str in ["uint8", "u1"]:
        return (polars.UInt8, None, None)
    elif dtype_str in ["uint16", "u2"]:
        return (polars.UInt16, None, None)
    elif dtype_str in ["uint32", "u4"]:
        return (polars.UInt32, None, None)
    elif dtype_str in ["uint64", "u8"]:
        return (polars.UInt64, None, None)
    elif dtype_str in ["float32", "f4"]:
        return (polars.Float32, None, None)
    elif dtype_str in ["float64", "f8"]:
        return (polars.Float64, None, None)
    elif dtype_str == "bool":
        return (polars.Boolean, None, None)
    elif dtype_str.startswith("str") or dtype_str.startswith("u"):
        return (polars.String, None, None)
    elif dtype_str.startswith("bytes") or dtype_str.startswith("s"):
        return (polars.Binary, None, None)

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


def list_files(
    path: list[str]
    | list[pathlib.Path]
    | list[str | pathlib.Path]
    | str
    | pathlib.Path,
    extension: str | None = None,
    recursive: bool = False,
    include_directories: bool = False,
) -> list[pathlib.Path]:
    """Lists files in a directory, with glob pattern support.

    Parameters
    ----------
    path
        The path to list files in. Can be a string or a pathlib.Path object.
        Can include glob patterns.
    extension
        If provided, only files with this extension will be returned. Matched as
        a regular expression against the file name.
    recursive
        If True, directories will be searched recursively. Ignored if glob patterns
        are used, as they can already include recursive patterns (e.g., ``**/*``).
    include_directories
        If True, directories will be included in the returned list.

    Returns
    -------
    list[pathlib.Path]
        A list of Path objects representing the files that match the input path.

    """

    if isinstance(path, (str, pathlib.Path)):
        path = [path]

    # Start by getting all the files in the provided paths that match the path
    # glob pattern, or all files in the directory if no glob pattern is provided.
    path_tmp: list[pathlib.Path | str] = []
    for file in path:
        if is_glob_pattern(file):
            for matched_file in glob.glob(str(file)):
                if pathlib.Path(matched_file).is_file() or include_directories:
                    path_tmp.append(matched_file)
        else:
            path = pathlib.Path(file)
            if path.is_dir():
                files = path.glob("**/*") if recursive else path.glob("*")
                for matched_file in files:
                    if matched_file.is_file() or include_directories:
                        path_tmp.append(matched_file)
            elif path.is_file() or include_directories:
                path_tmp.append(path)

    paths = [pathlib.Path(ff) for ff in path_tmp]

    if extension is not None:
        if not extension.startswith("."):
            extension = "." + extension

        paths = [ff for ff in paths if re.search(rf"{extension}$", ff.name)]

    return paths


def get_current_time(format: str = "isot", timezone: str = "UTC") -> str:
    """Returns the current time as a string in the specified format and timezone."""

    if format.lower() != "isot":
        raise NotImplementedError(f"Time format '{format}' is not supported.")

    if timezone.upper() != "UTC":
        raise NotImplementedError(f"Timezone '{timezone}' is not supported.")

    now = datetime.datetime.now(datetime.timezone.utc)

    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def sink_to_file(
    lf: polars.LazyFrame,
    file_path: str | pathlib.Path,
    overwrite: bool = False,
) -> None:
    """Writes content to a file, optionally overwriting it.

    If the file exists, writes the lazy frame to a temporary file and then renames
    it to the target file path.

    """

    file_path = pathlib.Path(file_path)

    if file_path.exists() and not overwrite:
        raise FileExistsError("File already exists. Use overwrite to replace it.")

    tmp_file = file_path.with_suffix(file_path.suffix + ".tmp")
    if tmp_file.exists():
        raise FileExistsError("Temporary file already exists. Cannot sink.")

    if file_path.suffix == ".parquet":
        lf.sink_parquet(tmp_file)
    elif file_path.suffix == ".csv":
        lf.sink_csv(tmp_file)
    else:
        raise ValueError(f"Unsupported file extension: {file_path.suffix}")

    tmp_file.rename(file_path)
