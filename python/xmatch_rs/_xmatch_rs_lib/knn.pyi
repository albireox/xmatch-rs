#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-06-11
# @Filename: knn.pyi
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import numpy
import numpy.typing as npt

def match_sky(
    ref_coords: npt.ArrayLike[tuple[float, float]],
    match_coords: npt.ArrayLike[tuple[float, float]],
    k: int,
) -> tuple[list[npt.NDArray[numpy.uint64]], list[npt.NDArray[numpy.float64]]]:
    """Matches two sets of spherical coordinates using a k-nearest neighbors search.

    Parameters
    ----------
    ref_coords
        The first set of coordinates as a list of (RA, Dec) tuples in degrees. This is
        the "referece" catalogue of coordinates where the neighbors are searched.
    match_coords
        The second set of coordinates as a list of (RA, Dec) tuples in degrees.
        This is the set of coordinates for which the neighbors in ``ref_coords`` are
        searched.
    k
        The number of nearest neighbors to retrieve for each coordinate in
        ``match_coords``.

    Returns
    -------
    indices
        A tuple of two lists: the first list contains the indices of the neighbors in
        ``ref_coords`` for each coordinate in ``match_coords``, and the second list
        contains the corresponding distances in degrees. Each list has the same length
        as ``match_coords``, and each element is a Numpy array of length ``k``.

    """
    ...

def match_sky_radius(
    ref_coords: npt.ArrayLike[tuple[float, float]],
    match_coords: npt.ArrayLike[tuple[float, float]],
    radius: float,
) -> tuple[list[npt.NDArray[numpy.uint64]], list[npt.NDArray[numpy.float64]]]:
    """Matches two sets of spherical coordinates using a radius search.

    Parameters
    ----------
    ref_coords
        The first set of coordinates as a list of (RA, Dec) tuples in degrees. This is
        the "referece" catalogue of coordinates where the neighbors are searched.
    match_coords
        The second set of coordinates as a list of (RA, Dec) tuples in degrees.
        This is the set of coordinates for which the neighbors in ``ref_coords`` are
        searched.
    radius
        The search radius in degrees. For each coordinate in ``match_coords``,
        all neighbors in ``ref_coords`` that fall within this radius are retrieved.

    Returns
    -------
    indices
        A tuple of two lists: the first list contains the indices of the neighbors in
        ``ref_coords`` for each coordinate in ``match_coords``, and the second list
        contains the corresponding distances in degrees. Each list has the same length
        as ``match_coords``, and each element is a Numpy array of variable length
        depending on how many neighbors fall within the specified radius.

    """
    ...
