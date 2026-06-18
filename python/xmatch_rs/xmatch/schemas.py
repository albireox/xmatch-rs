#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-06-13
# @Filename: schemas.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import polars


__all__ = ["CATALOG_SCHEMA", "CATALOG_TO_TABLE_SCHEMA"]

CATALOG_SCHEMA = polars.Schema(
    {
        "catalogid": polars.UInt64,
        "healpix_29": polars.UInt64,
        "ra": polars.Float64,
        "dec": polars.Float64,
        "pmra": polars.Float64,
        "pmdec": polars.Float64,
        "epoch": polars.Float32,
        "added_by": polars.String,
        "added_on": polars.Datetime(time_unit="ms", time_zone="UTC"),
        "version": polars.String,
    }
)


CATALOG_TO_TABLE_SCHEMA = polars.Schema(
    {
        "catalogid": polars.UInt64,
        "healpix_29": polars.UInt64,
        "version": polars.String,
        "distance": polars.Float64,
        "best": polars.Boolean,
        "added_by_phase": polars.Int8,
        "added_on": polars.Datetime(time_unit="ms", time_zone="UTC"),
    }
)
