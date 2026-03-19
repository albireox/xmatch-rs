#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-03-18
# @Filename: healpix.pyi
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

def ra_dec_to_healpix(ra: float, dec: float, depth: int) -> int:
    """Converts RA and Dec to HEALPix pixel index.

    Parameters
    ----------
    ra
        Right Ascension in degrees.
    dec
        Declination in degrees.
    depth
        HEALPix depth/order (0 to 29).

    Returns
    -------
    int
        HEALPix pixel index corresponding to the given RA and Dec
        at the specified depth.

    """

    ...
