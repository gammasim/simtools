This repository contains simtools: a Python toolkit for configuring, running, and validating Monte Carlo simulations for CTAO using CORSIKA + sim_telarray. Agents should follow the conventions below to be productive.

## Core architecture and patterns

- Source lives under `src/simtools/**`. CLI entry points are thin wrappers in `src/simtools/applications/**` and are wired in `pyproject.toml [project.scripts]`. Put orchestration and logic in modules/classes, not in application scripts.
- Key domains (examples):
	- Model + DB: `simtools.model.*` and `database_scripts/**` for Mongo-backed parameter sets; use `initialize_simulation_models(...)` to get models.
	- Simulation runners: `simtools/corsika/**`, `simtools/simtel/**`, `simtools/ray_tracing/**` (e.g., `IncidentAnglesCalculator`) encapsulate preparation, execution, and parsing of outputs.
	- Visualization: `simtools/visualization/**` provides plotting functions that consume `astropy.table.QTable` with `astropy.units` columns.
- Data flows: applications configure models ➜ write config files ➜ run external binaries ➜ parse text/HDF5 outputs ➜ return domain tables (QTable) ➜ optional plots/summaries saved under `output_dir/**`.
- Units: use `astropy.units` for physical quantities. Convert when plotting/serializing.

## Developer workflows

- Environment: Python ≥ 3.11. Install extras for dev and tests: `pip install -e .[dev,tests]`.
- Lint/format: ruff + pylint configured in `pyproject.toml`. Prefer `logging` over `print`, `pathlib.Path` over `os.path`.
- Tests: `pytest -q` runs unit tests (see `tool.pytest.ini_options`). Tests live in `tests/unit_tests/` mirroring `src/simtools/` tree. Do not add tests under `src/simtools/applications/**` (excluded from coverage).
- Docs: Sphinx in `docs/`; changes go through Towncrier fragments in `docs/changes/` for releases.

## Project-specific conventions

- File I/O: never hardcode absolute paths; accept `Path` or directories and create subfolders as needed (e.g., `plots/`, `logs/`, `incident_angles/`).
- Logging: get a module-level logger; use concise messages and f-strings. Avoid printing. Warn and continue on partial data.
- Tables: return `QTable` with unit-bearing columns; example columns for ray tracing: `angle_incidence_focal`, `angle_incidence_primary`, `primary_hit_radius`, `primary_hit_x`, etc. Use meters and degrees.
- Parsing sim_telarray outputs: prefer header-driven column discovery with safe fallbacks. Guard against missing/short rows and non-numeric values; fill with `nan` where appropriate.
- Plotting: functions in `simtools.visualization.*` accept `dict[float, QTable]` keyed by off-axis angle and write PNGs to `output_dir/plots`. Use tight layout; avoid interactive backends.

## External integrations

- External binaries: CORSIKA and sim_telarray live outside the repo. Build commands are scripted in runner classes; do not shell out directly from tests. Respect environment variables and paths passed via config.
- Database: Mongo scripts in `database_scripts/**` help bootstrap local DBs. Code uses `pymongo` via model utilities.
- Containers: Dockerfiles under `docker/` with a dev image; see `docker/README.md` for example `docker run` commands.

## Examples and references

- Minimal CLI pattern: see `src/simtools/applications/plot_array_layout.py` after refactor; only parse args and call into `simtools.layout.array_layout_utils`.
- Ray-tracing workflow: `src/simtools/ray_tracing/incident_angles.py` computes angles and hit geometry; plotting in `src/simtools/visualization/plot_incident_angles.py` consumes its QTables to produce histograms and 2D heatmaps.
- Scripts mapping: `pyproject.toml [project.scripts]` lists all applications and their module entry points.

## Guidelines

- Always use semantic versions (e.g., `1.0.0`, no leading `v`).
- Do not start replies with “this is the final outcome”.
- Avoid trivial comments; keep code comments minimal and meaningful.
- Use `pathlib`, `logging`, and f-strings; avoid hardcoded paths.
- Do not use Python type hinting in new code (avoid annotations in function signatures and variables).
- Use numpydoc-style docstrings for public APIs (Parameters/Returns sections) consistent with project docs.

## Tests

- Place unit tests under `./tests/unit_tests/` mirroring the package structure.
- Use simple test functions (no classes). Ensure proper indentation and imports at top.
- Check `tests/conftest.py` for fixtures and pytest configuration.
