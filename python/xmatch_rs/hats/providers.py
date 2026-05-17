#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-04-06
# @Filename: providers.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import re
import uuid
from functools import partial

from typing import Callable

import polars


___all__ = ["get_hats_file_provider"]


def _hats_provider(
    provider_args: polars.FileProviderArgs,
    column_name: str,
    order: int,
    use_uuid: bool = False,
) -> pathlib.Path:
    """A file provider for the HATS specification."""

    index = provider_args.index_in_partition
    keys = provider_args.partition_keys

    if column_name not in keys.columns:
        raise ValueError(f"Column '{column_name}' not found in partition keys.")

    keys_values = keys[column_name].to_list()
    if (n_val := len(keys_values)) != 1:
        raise ValueError(f"Expected one value in column {column_name!r}, got {n_val}.")

    pixel = keys_values[0]

    dir_n = (pixel // 10_000) * 10_000

    part = f"part{index:08d}.parquet"

    if not use_uuid:
        return pathlib.Path(f"Norder={order}/Dir={dir_n}/Npix={pixel}/{part}")
    else:
        uuid_part = uuid.uuid4().hex.lower()
        file_ = f"{uuid_part}-{index:02d}.parquet"
        return pathlib.Path(f"Norder={order}/Dir={dir_n}/Npix={pixel}/{file_}")


def get_hats_file_provider(
    order: int | str,
    column_name: str | None = None,
    use_uuid: bool = False,
) -> Callable[[polars.FileProviderArgs], pathlib.Path]:
    """Returns a file provider for the HATS specification.

    Parameters
    ----------
    order
        The order of the HEALPix grid.
    column_name
        The name of the column to use for the provider. If `None`, the column name will
        be inferred from the HEALPix index.
    use_uuid
        Whether to use a UUID in the file name. If ``False``, the file name will be
        deterministic based on the partition index. If ``True``, a random UUID will be
        used in the file name to ensure uniqueness.

    Returns
    -------
    file_provider
        A file provider that can be used with the HATS specification.

    """

    if isinstance(order, str):
        try:
            order = int(order)
        except ValueError:
            assert isinstance(order, str)

            if match := re.match(r"_?healpix_(\d+)", order, re.IGNORECASE):
                order = int(match.group(1))
            else:
                raise ValueError(f"Could not parse order from string {order!r}.")

    if column_name is None:
        column_name = f"_healpix_{order}"

    _provider = partial(
        _hats_provider,
        column_name=column_name,
        order=order,
        use_uuid=use_uuid,
    )

    return _provider
