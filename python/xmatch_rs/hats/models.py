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
from pydantic import (
    AnyUrl,
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from xmatch_rs.schema import Schema, SchemaModel


__all__: list[str] = ["ConvertToHATSModel", "PhaseModel", "HATSPropertiesModel"]


class ConvertToHATSModel(BaseModel):
    """Model representing the conversion to HATS format."""

    catalogue_model: Annotated[
        SchemaModel,
        Field(description="The schema to use for the conversion."),
    ]

    object_path: Annotated[
        pathlib.Path,
        Field(
            description="The path where the output object catalogue will be "
            "created inside the HATS collection."
        ),
    ]

    complete: Annotated[
        bool,
        Field(description="Whether the conversion is complete."),
    ] = False

    reference_order: Annotated[
        int,
        Field(description="The reference HEALPix order."),
    ] = 29

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

    def get_phase(self, phase_name: str) -> PhaseModel | None:
        """Returns the model for a given phase name."""

        for phase in self.phases:
            if phase.name == phase_name:
                return phase

        return None

    def get_schema(self) -> Schema:
        """Returns the schema of the conversion model."""

        return Schema(self.catalogue_model.model_dump())

    def add_phase(
        self,
        phase: PhaseModel | None = None,
        write: bool = True,
        add_after: str | None = None,
        **kwargs,
    ) -> PhaseModel:
        """Adds a phase to the conversion model."""

        if phase is None:
            phase = PhaseModel(**kwargs)

        if add_after is not None:
            for ii, existing_phase in enumerate(self.phases):
                if existing_phase.name == add_after:
                    self.phases.insert(ii + 1, phase)
                    break
            else:
                raise ValueError(f"Phase {add_after!r} not found in conversion model.")
        else:
            self.phases.append(phase)

        if write:
            self.write()

        return phase

    def get_next_phase(self) -> PhaseModel | None:
        """Returns the next phase to run, i.e., the first phase that is not complete."""

        for phase in self.phases:
            if not phase.complete:
                return phase

        return None

    def update_phase(
        self,
        phase_name: str,
        complete: bool | None = None,
        write: bool = True,
    ) -> None:
        """Updates the progress model for a given phase name."""

        phase = self.get_phase(phase_name)
        if phase is None:
            raise ValueError(f"Phase {phase_name} not found in conversion model.")

        if complete is not None:
            phase.complete = complete

        if write:
            self.write()

    def write(self) -> None:
        """Writes the model to a ``_convert.yaml`` file in the output path."""

        output_file = self.object_path / "_convert.yaml"
        lock = FileLock(self.object_path / "_convert.yaml.lock")

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
        Field(description="The initial HEALPix order."),
    ] = None

    healpix_order_1: Annotated[
        int | None,
        Field(description="The final HEALPix order."),
    ] = None

    max_rows_per_pixel: Annotated[
        int | None,
        Field(description="The maximum number of rows per pixel."),
    ] = None

    max_kilobytes_per_pixel: Annotated[
        int | None,
        Field(description="The maximum number of kilobytes per pixel."),
    ] = None

    max_rows_per_file: Annotated[
        int | None,
        Field(description="The maximum number of rows per file."),
    ] = None

    max_kilobytes_per_file: Annotated[
        int | None,
        Field(description="The maximum number of kilobytes per file."),
    ] = None

    keep_lower_order_directories: Annotated[
        bool,
        Field(description="Keep the lower order directories after partitioning."),
    ] = False


class HATSPropertiesModel(BaseModel):
    """Model for HATS properties file based on IVOA HATS standard Table 1."""

    header: Annotated[
        str | None,
        Field(description="Header for the properties file"),
    ] = None

    # Collection identification
    obs_collection: Annotated[
        str,
        Field(description="Short name of original dataset (e.g., 2MASS)"),
    ]
    creator_did: Annotated[
        str | None,
        Field(description="Unique ID of the HATS in IVOID format. "),
    ] = None
    addendum_did: Annotated[
        str | None,
        Field(
            description="Creator_did of any added data if content "
            "has been added after initial catalogue creation",
        ),
    ] = None
    publisher_id: Annotated[
        str | None,
        Field(description="Unique ID of the HATS publisher in IVOID format."),
    ] = None

    # Data product information
    dataproduct_type: Annotated[
        Literal["object", "margin", "association", "index", "collection"],
        Field(description="Type of HATS product"),
    ]
    obs_title: Annotated[
        str | None,
        Field(description="Data set title (e.g., HST F110W observations)"),
    ] = None
    obs_description: Annotated[
        str | None,
        Field(description="Longer free text description of the dataset"),
    ] = None
    data_ucd: Annotated[
        str | None,
        Field(description="UCD describing data contents"),
    ] = None

    # Spatial and coordinate information
    hats_col_ra: Annotated[
        str | None,
        Field(description="Column name of the RA coordinate"),
    ] = None
    hats_col_dec: Annotated[
        str | None,
        Field(description="Column name of the Dec coordinate"),
    ] = None
    hats_frame: Annotated[
        Literal["equatorial", "galactic", "ecliptic"] | None,
        Field(description="Coordinate frame reference"),
    ] = None
    hats_coordinate_epoch: Annotated[
        str | None,
        Field(description="Measurement epoch for default RA and Dec columns"),
    ] = None

    # HATS structure
    hats_nrows: Annotated[
        int | None,
        Field(description="Number of rows of the HATS catalogue"),
    ] = None
    hats_order: Annotated[
        int | None,
        Field(description="Deepest HATS order"),
    ] = None
    hats_max_rows: Annotated[
        int | None,
        Field(
            description="Maximum number of rows per file before breaking "
            "into 4 new files at higher order",
        ),
    ] = None
    hats_cols_sort: Annotated[
        str | None,
        Field(description="Columns used to sort the data (space-delimited)"),
    ] = None
    hats_cols_survey_id: Annotated[
        str | None,
        Field(description="Primary key from the original survey data"),
    ] = None
    hats_cols_default: Annotated[
        str | None,
        Field(description="Columns to read by default (space-delimited)"),
    ] = None
    npix_suffix: Annotated[
        str | None,
        Field(description="File suffix for leaf files (e.g., '.parquet' or '/')"),
    ] = None

    # Margin cache (for margin tables)
    hats_margin_threshold: Annotated[
        float | None,
        Field(description="Threshold for finding points within margin (arcseconds)"),
    ] = None

    # Index table
    hats_index_column: Annotated[
        str | None,
        Field(description="Column that is indexed over"),
    ] = None
    hats_index_extra_column: Annotated[
        str | None,
        Field(description="Extra columns carried through with the index"),
    ] = None

    # Association table
    hats_col_assn_primary: Annotated[
        str | None,
        Field(description="Column name for the primary (left) side of the join"),
    ] = None
    hats_col_assn_primary_assn: Annotated[
        str | None,
        Field(description="Column in the association table for primary side"),
    ] = None
    hats_col_assn_join: Annotated[
        str | None,
        Field(description="Column name for the joining (right) side"),
    ] = None
    hats_col_assn_join_assn: Annotated[
        str | None,
        Field(description="Column in the association table for join side"),
    ] = None
    hats_assn_join_table_url: Annotated[
        AnyUrl | None,
        Field(description="URL to join table with original survey data"),
    ] = None
    hats_assn_leaf_files: Annotated[
        bool | None,
        Field(description="Whether association table contains leaf files"),
    ] = None

    # Supplemental table references
    hats_primary_table_url: Annotated[
        AnyUrl | None,
        Field(description="URL to the primary table with original survey data"),
    ] = None
    hats_progenitor_url: Annotated[
        AnyUrl | None,
        Field(description="URL to an associated progenitor HATS catalogue"),
    ] = None
    hats_service_url: Annotated[
        AnyUrl | None,
        Field(description="HATS access URL"),
    ] = None

    # Collection-specific
    all_margins: Annotated[
        str | None,
        Field(
            description="Space-delimited list of subdirectories "
            "containing margin caches"
        ),
    ] = None
    default_margin: Annotated[
        str | None,
        Field(description="Default margin cache to use for cross-matching"),
    ] = None
    all_indexes: Annotated[
        str | None,
        Field(description="Space-delimited map of indexed field to subdirectories"),
    ] = None
    default_index: Annotated[
        str | None,
        Field(description="Default index field for ID searches"),
    ] = None

    # Metadata and provenance
    hats_builder: Annotated[
        str | None,
        Field(description="Name and version of building tool"),
    ] = None
    hats_creation_date: Annotated[
        str | None,
        Field(description="HATS creation date in ISO 8601 format (YYYY-mm-ddTHH:MMZ)"),
    ] = None
    hats_release_date: Annotated[
        str | None,
        Field(description="Last HATS update date in ISO 8601 format"),
    ] = None
    hats_version: Annotated[
        str | None,
        Field(description="HATS version number (e.g., 0.1)"),
    ] = None
    hats_status: Annotated[
        str | None,
        Field(
            description="HATS status (space-delimited: private/public, "
            "main/mirror/partial, clonable/unclonable/clonableOnce)",
        ),
    ] = None
    hats_copyright: Annotated[
        str | None,
        Field(description="Copyright mention associated with dataset"),
    ] = None
    hats_creator: Annotated[
        str | None,
        Field(description="Institute or person who built the HATS catalogue"),
    ] = None
    hats_estsize: Annotated[
        int | None,
        Field(description="HATS size estimation in KB"),
    ] = None

    # Observable characteristics
    obs_regime: Annotated[
        Literal[
            "Radio",
            "Millimeter",
            "Infrared",
            "Optical",
            "UV",
            "EUV",
            "X-ray",
            "Gamma-ray",
        ]
        | None,
        Field(description="General wavelength regime"),
    ] = None
    obs_ack: Annotated[
        str | None,
        Field(description="Acknowledgment"),
    ] = None
    obs_copyright: Annotated[
        str | None,
        Field(description="Copyright associated with original data"),
    ] = None
    obs_copyright_url: Annotated[
        AnyUrl | None,
        Field(description="URL to copyright information"),
    ] = None

    # References and provenance
    bib_reference: Annotated[
        str | None,
        Field(description="Bibliographic reference"),
    ] = None
    bib_reference_url: Annotated[
        AnyUrl | None,
        Field(description="URL to bibliographic reference"),
    ] = None
    prov_progenitor: Annotated[
        str | None,
        Field(description="Provenance of the original data"),
    ] = None

    # Time coverage
    t_min: Annotated[
        float | None,
        Field(description="Start time of observations (MJD)"),
    ] = None
    t_max: Annotated[
        float | None,
        Field(description="Stop time of observations (MJD)"),
    ] = None

    # Sky coverage
    moc_sky_fraction: Annotated[
        float | None,
        Field(description="Fraction of sky covered by MOC (0 to 1)"),
    ] = None

    class Config:
        """Pydantic config."""

        extra = "allow"  # Allow additional fields for forward compatibility

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self.model_dump().items())
        return f"<HATSPropertiesModel ({fields})>"

    @model_serializer(mode="wrap")
    def serialize_model(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        """Custom serializer to remove null fields."""

        serialized = handler(self)
        serialized = {k: v for k, v in serialized.items() if v is not None}

        return serialized

    def write(self, output_path: pathlib.Path | str) -> None:
        """Writes the properties to a YAML file in the output path."""

        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            if self.header:
                header = self.header.strip().strip("#").strip()
                f.write(f"# {header}\n")

            for key, value in self.model_dump().items():
                if key == "header":
                    continue
                if value is not None:
                    f.write(f"{key}={str(value)}\n")

    @classmethod
    def read(cls, input_path: pathlib.Path | str) -> HATSPropertiesModel:
        """Reads the properties from a YAML file in the input path."""

        input_path = pathlib.Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file {input_path} does not exist.")

        if not input_path.is_file():
            raise ValueError(f"Input path {input_path} is not a file.")

        with input_path.open("r", encoding="utf-8") as f:
            first_line: bool = True

            fields: dict[str, str] = {}

            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue

                if line.startswith("#"):
                    if first_line:
                        fields["header"] = line.strip("#").strip()
                elif "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    fields[key] = value
                else:
                    raise ValueError(f"Invalid line in properties file: {line}")

                first_line = False

        return cls.model_validate(fields)
