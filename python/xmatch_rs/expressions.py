#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-03-18
# @Filename: expressions.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from pathlib import Path

from typing import TypeAlias, Union

import polars
from polars.datatypes import DataType, DataTypeClass
from polars.plugins import register_plugin_function


__all__: list[str] = ["get_healpix"]


IntoExprColumn: TypeAlias = Union[polars.Expr, str, polars.Series]
PolarsDataType: TypeAlias = Union[DataType, DataTypeClass]

LIB = Path(__file__).parent


def get_healpix(ra: IntoExprColumn, dec: IntoExprColumn, *, depth: int) -> polars.Expr:
    return register_plugin_function(
        args=[ra, dec],
        plugin_path=LIB,
        function_name="get_healpix",
        is_elementwise=True,
        kwargs={"depth": depth},
    )
