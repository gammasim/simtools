# Architecture Rules

These rules apply to new and changed code. Reviewers refer to them.

## Layers and imports

The package is organized in layers. A module may import only from its own or a
lower layer. Library code never imports `simtools.applications`, and
production code never imports test helpers.

Import-linter enforces the package layers, and Ruff checks private cross-module
imports. Three lists are ratchets: reduce them as the architecture cleanup
proceeds and never add entries or raise the count. They are the
`ignore_imports` of `[tool.importlinter]` and the `ignore_names` of
`[tool.vulture]` in `pyproject.toml`, and the `--max` value of the
`internal-patch-count` hook in `.pre-commit-config.yaml` (patches of
`simtools.*` internals in unit tests).

```text
5  applications/            thin CLI wrappers
4  workflows                simulate, production, science_tests, reporting, derive_*
3  simulation engines       corsika, simtel, ray_tracing, camera, job_execution
2  model                    model (site/telescope/array), model_repository, layout
1  data                     io, data_model (schemas, metadata), sim_events readers
0  foundation               utils, constants, names, units
   visualization            may import 0-2 only; receives data, never computes it
```

Plotting modules receive arrays, tables or dataclasses and draw them.
Rebinning, statistics and comparison metrics belong in the module that
computes them, so they can be tested without matplotlib.

## Adding is data, not code

Prefer data over code for anything that is expected to grow:

- A new model parameter is a schema in `schemas/model_parameters/` plus model
  data. Simulator-specific names and formats belong in schema fields
  (`simulation_software`, `internal_parameter_name`). Add code only for a new
  output format.
- A shared file product (suffix, default name) is an entry in
  `resources/file_types.yml`.
- A simple line or histogram plot is a `plot_tables` configuration.

## Extension points

Each axis where simtools grows has one extension point:

| Axis | Extension point |
| --- | --- |
| Model parameter | schema YAML in `schemas/model_parameters/` |
| Simulation configuration writer and result reader | see [Simulation formats](simulation_formats.md) |
| Ray-tracing backend | `ray_tracing/backends/interface.py` |
| Event-file format | `sim_events/formats/readers.py` |
| Execution backend | `job_execution/backends/registry.py` |
| Plot | `plot_tables` and `plot_configuration.metaschema.yml` |
| Application | `ApplicationDefinition` plus the `[project.scripts]` entry |

New extension points use a small `typing.Protocol` and a module-level dict
mapping names to implementations (with lazy imports for optional
dependencies). Do not add `register_*` functions, factories or plug-in
machinery outside these axes. Engine-specific vocabulary (CORSIKA keywords,
sim_telarray option names) stays inside the engine's package; other code uses
physics terms.

## Public API

A name is public when it is documented in the API reference and does not start
with an underscore. A leading underscore means private to its module: do not
import it from another module; make it public or move it instead. Use
`__all__` only in package `__init__.py` files that re-export names; do not
define per-module `__all__` lists.

Before v1.0, API changes do not keep compatibility aliases. Remove or rename
names directly and list each change in the changelog.

## Test policy

- Unit tests do not use checked-in or downloaded test resources. Generate
  inputs in `tmp_test_directory` from the packaged schemas.
- Mock only true external boundaries: external programs (CORSIKA,
  sim_telarray, container engines, HTCondor), network access, model sources
  (through the shared model-reader fixtures in `tests/unit_tests/conftest.py`),
  and time, UUID or randomness when determinism is needed. Use
  `monkeypatch.setenv` for environment variables.
- Do not add patches of other `simtools.*` internals. Test public behavior;
  shape code as plain functions on arrays, tables or dataclasses so that it
  needs no mocks.
- File-format compatibility with real CORSIKA and sim_telarray output belongs
  in integration tests.
