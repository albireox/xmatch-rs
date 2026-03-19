#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-03-17
# @Filename: __init__.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import rich.console

from sdsstools import get_logger


console = rich.console.Console()

log = get_logger("xmatch_rs", use_rich_handler=True)
log.rich_console = console
