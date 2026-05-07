#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-04-07
# @Filename: models.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import Annotated, Literal

import yaml
from filelock import FileLock
from pydantic import BaseModel, Field

from xmatch_rs.schema import Schema, SchemaModel


__all__: list[str] = ["ConvertToHATSModel", "PhaseModel"]


class ConvertToHATSModel(BaseModel):
    """Model representing the conversion to HATS format."""

    dataset_schema: Annotated[
        SchemaModel,
        Field(description="The schema to use for the conversion."),
    ]

    output_path: Annotated[
        pathlib.Path,
        Field(description="The path where the converted file will be saved."),
    ]

    complete: Annotated[
        bool,
        Field(description="Whether the conversion is complete."),
    ] = False

    phases: Annotated[
        list[PhaseModel],
        Field(
            description="Phases of the conversion to HATS format.",
            default_factory=list,
        ),
    ]

    @classmethod
    def read(cls, output_path: pathlib.Path | str) -> ConvertToHATSModel:
        """Reads the ``_convert.yaml`` file in the output path."""

        output_path = pathlib.Path(output_path)
        input_file = output_path / "_convert.yaml"
        if not input_file.exists():
            raise FileNotFoundError(f"No conversion model found at {input_file}.")

        with input_file.open("r") as f:
            convert_dict = yaml.safe_load(f)

        convert_model = ConvertToHATSModel.model_validate(convert_dict)
        return convert_model

    def get_phase_names(self) -> list[str]:
        """Returns a list of the names of the phases in the conversion model."""

        return [phase.name for phase in self.phases]

    def get_phase(self, phase_name: str) -> PhaseModel:
        """Returns the model for a given phase name."""

        for phase in self.phases:
            if phase.name == phase_name:
                return phase

        raise ValueError(f"Phase {phase_name} not found in conversion model.")

    def get_schema(self) -> Schema:
        """Returns the schema of the conversion model."""

        return Schema(self.dataset_schema.model_dump())

    def add_phase(
        self,
        phase: PhaseModel | None = None,
        write: bool = True,
        **kwargs,
    ) -> PhaseModel:
        """Adds a phase to the conversion model."""

        if phase is None:
            phase = PhaseModel(**kwargs)

        self.phases.append(phase)

        if write:
            self.write()

        return phase

    def update_phase(
        self,
        phase_name: str,
        complete: bool | None = None,
        write: bool = True,
    ) -> None:
        """Updates the progress model for a given phase name."""

        phase = self.get_phase(phase_name)
        if complete is not None:
            phase.complete = complete

        if write:
            self.write()

    def write(self) -> None:
        """Writes the model to a ``_convert.yaml`` file in the output path."""

        output_file = self.output_path / "_convert.yaml"
        lock = FileLock(self.output_path / "_convert.yaml.lock")

        with lock:
            with output_file.open("w") as f:
                f.write(yaml.dump(self.model_dump(mode="json"), sort_keys=False))


class PhaseModel(BaseModel):
    """Model representing the progress of a single phase of the conversion."""

    name: Annotated[
        str,
        Field(description="The name of the phase."),
    ]

    type: Annotated[
        Literal["healpix_partitioning", "rechunk"],
        Field(description="The type of the phase."),
    ] = "healpix_partitioning"

    description: Annotated[
        str,
        Field(description="A description of the phase."),
    ] = ""

    complete: Annotated[
        bool,
        Field(description="Whether the phase is complete."),
    ] = False

    healpix_order_0: Annotated[
        int | None,
        Field(description="The initial HEALPix order, if applicable."),
    ] = None

    healpix_order_1: Annotated[
        int | None,
        Field(description="The final HEALPix order, if applicable."),
    ] = None

    max_rows_per_pixel: Annotated[
        int | None,
        Field(description="The maximum number of rows per pixel, if applicable."),
    ] = None

    keep_lower_order_directories: Annotated[
        bool,
        Field(description="Keep the lower order directories after partitioning."),
    ] = False
