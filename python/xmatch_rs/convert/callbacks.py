#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-03-19
# @Filename: callbacks.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import polars


__all__ = ["callbacks", "exprs", "lowercase_column_names"]


def lowercase_column_names(lf: polars.LazyFrame) -> polars.LazyFrame:
    """A callback function that lowercases all column names in a lazy frame."""

    return lf.rename(str.lower)


def hours_to_degrees(column: str) -> polars.Expr:
    """A callback function that converts a column from hours to degrees."""

    return polars.col(column) * 15.0


callbacks = {
    "lowercase_column_names": lowercase_column_names,
}

exprs = {
    "hours_to_degrees": hours_to_degrees,
}
