#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-04-13
# @Filename: types.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import Any, Callable, Literal

import polars


DATA_PATH_T = (
    list[str] | list[pathlib.Path] | list[str | pathlib.Path] | str | pathlib.Path
)

LOW_MEMORY_STRATEGY_OPTIONS_T = Literal[
    "data_page_size",
    "streaming_chunk_size",
]

LOW_MEMORY_STRATEGY_T = (
    LOW_MEMORY_STRATEGY_OPTIONS_T
    | list[LOW_MEMORY_STRATEGY_OPTIONS_T]
    | dict[LOW_MEMORY_STRATEGY_OPTIONS_T, Any]
)

CALLBACK_FUNC_T = Callable[[polars.LazyFrame], polars.LazyFrame]

APPLY_CALLBACK_T = CALLBACK_FUNC_T | str

FILE_PATH_PROVIDER_T = Callable[[polars.FileProviderArgs], pathlib.Path]
