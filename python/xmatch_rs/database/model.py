#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2026-10-15
# @Filename: model.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import inspect
from dataclasses import dataclass

from typing import Annotated, Any, Type

from pydantic import create_model
from pydantic_core import PydanticUndefined
from sqlmodel import Field, SQLModel, create_engine
from sqlmodel import inspect as sql_inspect

from xmatch_rs.schema import Schema


@dataclass(frozen=True)
class ColumnMetadata:
    """Metadata for a column in a database table."""

    column_description: str | None = None
    column_unit: str | None = None
    polars_dtype: str | None = None


class SQLModelWithMetadata(SQLModel):
    """A SQLModel class that includes column metadata."""

    @classmethod
    def get_column_metadata(cls, column_name: str) -> ColumnMetadata | None:
        """Returns the metadata for a given column."""

        field_info = cls.model_fields.get(column_name)
        if field_info is None:
            return None

        column_metadata = field_info.metadata[1]
        return column_metadata


def sqlalchemy_type_to_python_type(sqlalchemy_type: Any) -> type:
    """Converts a SQLAlchemy type to a Python type.

    Parameters
    ----------
    sqlalchemy_type
        The SQLAlchemy type to convert.

    Returns
    -------
    type
        The corresponding Python type.

    """

    if sqlalchemy_type.__class__.__name__ == "INTEGER":
        return int
    elif sqlalchemy_type.__class__.__name__ == "BIGINT":
        return int
    elif sqlalchemy_type.__class__.__name__ == "SMALLINT":
        return int
    elif sqlalchemy_type.__class__.__name__ == "VARCHAR":
        return str
    elif sqlalchemy_type.__class__.__name__ == "TEXT":
        return str
    elif sqlalchemy_type.__class__.__name__ == "FLOAT":
        return float
    elif sqlalchemy_type.__class__.__name__ == "REAL":
        return float
    elif sqlalchemy_type.__class__.__name__ == "DOUBLE_PRECISION":
        return float
    elif sqlalchemy_type.__class__.__name__ == "BOOLEAN":
        return bool
    else:
        raise ValueError(f"Unsupported SQLAlchemy type: {sqlalchemy_type}")


def get_model_from_database(
    database_uri: str,
    schema: str,
    table_name: str,
    class_name: str | None = None,
    catalogue_schema: Schema | None = None,
) -> Type[SQLModel]:
    """Returns a SQLModel model class for a given database table.

    Parameters
    ----------
    database_uri
        The URI of the database.
    schema
        The schema of the table.
    table_name
        The name of the table.
    class_name
        The name of the class to create. If None, the class name will be generated
        from the table name.

    Returns
    -------
    Type[SQLModel]
        A SQLModel model class for the given table.

    """

    engine = create_engine(database_uri)
    inspector = sql_inspect(engine)

    columns = inspector.get_columns(table_name, schema=schema)
    primary_key = inspector.get_pk_constraint(table_name, schema=schema)

    indices = inspector.get_indexes(table_name, schema=schema)
    index_columns = [col for index in indices for col in index["column_names"]]

    fields: dict[str, Any] = {
        "__table_args__": (dict, {"schema": schema, "extend_existing": True}),
        "__tablename__": (str, table_name),
    }

    if class_name is None:
        class_name = "".join(word.capitalize() for word in table_name.split("_"))

    for column in columns:
        column_name = column["name"]
        column_ptype = sqlalchemy_type_to_python_type(column["type"])
        is_primary_key = column_name in primary_key.get("constrained_columns", [])
        is_unique = column.get("unique", False)
        is_nullable = column.get("nullable", True)

        if is_nullable:
            column_ptype = column_ptype | None

        if catalogue_schema is not None:
            schema_column = catalogue_schema.get_column(
                column_name,
                use_database_name=True,
            )
            if schema_column is None:
                raise ValueError(
                    f"Column {column_name!r} not found in catalogue schema."
                )

            description = schema_column.description
            unit = schema_column.unit
            polars_dtype = schema_column.get_polars_dtype()

            column_metadata = ColumnMetadata(
                column_description=description,
                column_unit=unit,
                polars_dtype=str(polars_dtype),
            )
        else:
            column_metadata = ColumnMetadata()

        fields[column_name] = Annotated[
            column_ptype,  # type: ignore
            Field(
                primary_key=is_primary_key,
                nullable=is_nullable,
                unique=is_unique,
                sa_type=column["type"],
                index=column_name in index_columns,
            ),
            column_metadata,
        ]

    TableModel = create_model(
        class_name,
        __base__=SQLModelWithMetadata,
        __cls_kwargs__={"table": True},
        **fields,
    )

    return TableModel


def none_undefined(value: Any) -> Any:
    """Returns None if the value is ``pydantic.PydanticUndefined``."""

    if value is PydanticUndefined:
        return None

    return value


def print_model(model: type[SQLModel]):
    """Prints a SQLModel class as Python code."""

    table_args = getattr(model, "__table_args__", None)

    # Imports
    print("from dataclasses import dataclass")
    print()
    print("from typing import Annotated")
    print()
    print("import sqlalchemy")
    print("from sqlmodel import Field, SQLModel")
    print()
    print()

    print(inspect.getsource(ColumnMetadata))
    print()

    print(inspect.getsource(SQLModelWithMetadata))
    print()

    print(f"class {model.__name__}(SQLModelWithMetadata, table=True):")
    print(f"    __tablename__ = '{model.__tablename__}'")
    if table_args is not None:
        print(f"    __table_args__ = {table_args}")

    print()

    for name, field_info in model.model_fields.items():
        if name.startswith("__"):
            continue

        annotation = field_info.annotation
        if isinstance(annotation, type):
            annotation = annotation.__name__

        field_db_metadata = field_info.metadata[0]
        column_metadata = field_info.metadata[1]

        sa_type_class = field_db_metadata.sa_type.__class__.__name__

        print(f"    {name}: Annotated[{annotation}, {column_metadata}] = Field(")
        print(f"        sa_type=sqlalchemy.types.{sa_type_class},")
        print(f"        primary_key={field_db_metadata.primary_key},")
        print(f"        nullable={field_db_metadata.nullable},")
        print(f"        index={field_db_metadata.index},")
        print("    )")
