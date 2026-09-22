# AGENTS.md

## Mission

This repository implements astronomical catalogue cross-matching with a high-performance Rust core and a Python interface. The package also implements tools to convert from different types of files (Parquet, FITS, CSV) to the [HATS](https://www.ivoa.net/documents/Notes/HATS/20250822/NOTE-hats-ivoa-1.0-20250822.html) format, as well as utilities to manage database load.

Overall, user-facing functionality is provided through the Python interface and CLI, while performance-critical operations are handled by the Rust core.

Keep changes small, correct, and consistent with the existing architecture. Prefer the least invasive fix that preserves the public API and behaviour of the library.

## Project structure

- `src/` — Rust implementation of the matching engine and supporting utilities.
- `python/xmatch_rs/` — Python package, CLI, conversion utilities, and schema helpers.
- `python/xmatch_rs/conversion/` — Tools to convert from different file formats (Parquet, FITS, CSV) to Parquet.
- `python/xmatch_rs/hats/` — Tools and utilities related to the HATS format, including converting datasets to HATS collections.
- `python/xmatch_rs/xmatch/` — Runs the cross-matching engine and provides related utilities.
- `python/xmatch_rs/cli.py` — Command-line interface for interacting with the package.
- `python/xmatch_rs/database/` — Database management and loading utilities.
- `tests/` — regression tests for Python-facing functionality.
- `docs/sphinx/` — Sphinx documentation.
- `Cargo.toml` and `pyproject.toml` — Rust and Python configuration.

## Working rules

- Prefer Rust for performance-critical or computationally heavy logic.
- Prefer Python for CLI glue, user-facing configuration, schema validation, and convenience wrappers.
- Preserve backwards compatibility unless a task explicitly requires a breaking change.
- Keep public interfaces and CLI behaviour stable unless the issue requires otherwise.
- Avoid unrelated refactors or broad cleanup in the same patch.
- Add or update tests for any bug fix or feature change that affects runtime behaviour.
- Match the surrounding style and naming conventions of the file being edited.

## Dependencies and tooling

This project is configured for modern Python and Rust tooling:

- Python: >= 3.12
- Rust: standard project toolchain via Cargo
- Python package management: `uv` is the preferred workflow for dev/test commands
- Linting: `ruff`
- Testing: `pytest`
- Build for Python bindings: `maturin`

## Validation commands

Run the smallest relevant validation for the scope of the change. Typical commands:

- `uv run pytest`
- `uv run ruff check .`
- `cargo test`
- `uv run maturin develop` when the Rust/Python extension needs to be rebuilt locally
- `uv run sphinx-build -b html docs/sphinx docs/sphinx/_build/html` for documentation changes

## Before finishing

- Verify the relevant checks pass before claiming completion.
- If the change touches both Rust and Python surfaces, run the appropriate Rust and Python validation.
- Summarise the verification commands and outcomes in the final response.
- Keep commit messages and PR summaries focused on the behaviour that changed.

## Communication expectations

- Be explicit about assumptions when something is ambiguous.
- Call out risk or edge cases when a fix changes observable behaviour.
- If a task is blocked by missing context, ask for the required information instead of guessing.
