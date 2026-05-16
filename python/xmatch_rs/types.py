#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-04-13
# @Filename: types.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import Callable

import polars


DATA_PATH_T = (
    list[str] | list[pathlib.Path] | list[str | pathlib.Path] | str | pathlib.Path
)

CALLBACK_FUNC_T = Callable[[polars.LazyFrame], polars.LazyFrame]

APPLY_CALLBACK_T = CALLBACK_FUNC_T | str

FILE_PATH_PROVIDER_T = Callable[[polars.FileProviderArgs], pathlib.Path]
