#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-03-17
# @Filename: __init__.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import rich.console

from sdsstools import get_logger, get_package_version


NAME = "xmatch_rs"

console = rich.console.Console()

XMATCH_RS_ROOT = pathlib.Path(__file__).parent

__version__ = get_package_version(package_name=NAME)


from .hats import HATS
from .schema import Schema
from .xmatch import XMatch
