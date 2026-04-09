#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-04-07
# @Filename: models.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import Annotated

import yaml
from filelock import FileLock
from pydantic import BaseModel, Field

from xmatch_rs.schema import Schema, SchemaModel


__all__: list[str] = ["ConvertToHATSModel", "ProgressModel", "PhaseProgressModel"]


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

    progress: Annotated[
        ProgressModel,
        Field(
            description="Progress of the conversion to HATS format.",
            default_factory=ProgressModel,
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

    def get_phase(self, phase_name: str) -> PhaseProgressModel:
        """Returns the progress model for a given phase name."""

        for phase in self.progress.phases:
            if phase.name == phase_name:
                return phase

        raise ValueError(f"Phase {phase_name} not found in conversion model.")

    def get_schema(self) -> Schema:
        """Returns the schema of the conversion model."""

        return Schema(self.dataset_schema.model_dump())

    def update_phase(
        self,
        phase_name: str,
        tiles_processed: int | None = None,
        tiles_required: int | None = None,
        complete: bool | None = None,
        progress: float | None = None,
        compute_progress: bool = True,
        write: bool = True,
    ) -> None:
        """Updates the progress model for a given phase name."""

        phase = self.get_phase(phase_name)
        if tiles_processed is not None:
            phase.tiles_processed = tiles_processed
        if tiles_required is not None:
            phase.tiles_required = tiles_required
        if complete is not None:
            phase.complete = complete

        if progress is not None:
            phase.progress = progress
        elif compute_progress:
            if phase.complete:
                phase.progress = 100.0
            elif phase.tiles_required is not None and phase.tiles_processed is not None:
                phase.progress = (phase.tiles_processed / phase.tiles_required) * 100

        if write:
            self.write()

    def write(self) -> None:
        """Writes the model to a ``_convert.yaml`` file in the output path."""

        output_file = self.output_path / "_convert.yaml"
        lock = FileLock(self.output_path / "_convert.yaml.lock")

        with lock:
            with output_file.open("w") as f:
                f.write(yaml.dump(self.model_dump(mode="json"), sort_keys=False))


class ProgressModel(BaseModel):
    """Model representing the progress of the conversion."""

    complete: Annotated[
        bool,
        Field(description="Whether the conversion is complete."),
    ] = False

    phases: Annotated[
        list[PhaseProgressModel],
        Field(
            description="Progress of each phase of the conversion.",
            default_factory=list,
        ),
    ]


class PhaseProgressModel(BaseModel):
    """Model representing the progress of a single phase of the conversion."""

    name: Annotated[
        str,
        Field(description="The name of the phase."),
    ]

    description: Annotated[
        str,
        Field(description="A description of the phase."),
    ] = ""

    complete: Annotated[
        bool,
        Field(description="Whether the phase is complete."),
    ] = False

    progress: Annotated[
        float,
        Field(description="Progress of the phase as a percentage."),
    ] = 0.0

    tiles_required: Annotated[
        int | None,
        Field(description="The number of tiles to be processed in this phase."),
    ] = None

    tiles_processed: Annotated[
        int | None,
        Field(description="The number of tiles processed."),
    ] = None
