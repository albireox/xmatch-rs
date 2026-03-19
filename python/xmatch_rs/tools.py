#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-03-18
# @Filename: tools.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import polars


__all__: list[str] = ["lazy_frame_row_count", "get_polars_dtype_from_string"]


def lazy_frame_row_count(lf: polars.LazyFrame) -> int:
    """Returns the number of rows in a lazy frame."""

    return lf.select(polars.count()).collect().item()  # type: ignore


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
