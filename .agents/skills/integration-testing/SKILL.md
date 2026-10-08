---
name: integration-testing
description: >-
  Create, update, or debug simtools integration test YAML configs for
  application workflows in tests/integration_tests/config, including model
  sources, model-version handling, test-resource macros, documentation
  metadata, and output validation blocks.
---

# Integration Testing for simtools

Use this skill for integration test configuration files under
`tests/integration_tests/config/`. These tests run full `simtools-*`
applications from YAML and validate their outputs.

Follow `AGENTS.md` and
`docs/source/developer-guide/testing_integration.md`. This skill owns integration-specific
instructions; use the guide for repository-wide conventions and environment selection.
For exact mechanics, inspect
`tests/integration_tests/test_applications_from_config.py`,
`tests/integration_tests/conftest.py`, `tests/conftest.py`,
`src/simtools/testing/`, and
`src/simtools/schemas/application_workflow.metaschema.yml`.

## Model Source and Environment

Integration tests read simulation models from a filesystem checkout or a fixed Git
revision. Pass `--simulation_models_path`, or pass
`--simulation_models_git_path` and `--simulation_models_git_revision`. Filesystem
and Git sources cannot be configured together.

On DESY working-group servers, run integration tests in the published simtools
Apptainer development environment. From the repository checkout, start the
container and prepare the editable installation:

```bash
apptainer shell --writable-tmpfs \
  --bind "$PWD:/workdir/external" \
  docker://ghcr.io/gammasim/simtools-dev:latest
source /workdir/env/bin/activate
cd /workdir/external
pip install --no-build-isolation -e .
```

Apptainer downloads and caches the OCI image locally, normally under
`~/.apptainer/cache`. Set `APPTAINER_CACHEDIR` when that location is unsuitable.
Pin a published image tag instead of `latest` for reproducible runs.

## Config Shape

Use the latest version in `src/simtools/schemas/application_workflow.metaschema.yml`
for new configs (currently `0.6.0`). Preserve older versions unless migration is
part of the task. Versions `0.5.0` and `0.6.0` use `test_outputs` and `validations`;
legacy `test_output_files` and `test_simtel_cfg_files` belong to `0.4.0` only.

```yaml
---
applications:
- application: simtools-<app-name>
  configuration:
    model_version: 6.0.2
    output_path: simtools-output
    # application CLI options as YAML keys
  docs:
    title: Optional short title for rendered examples
    summary: Optional short summary for rendered examples
  integration_tests:
  - test_outputs:
    - file: relative/file/from/output_path.ext
      path_descriptor: output_path
  test_name: short_descriptive_case
schema_name: application_workflow.metaschema
schema_version: 0.6.0
```

Rules:

1. Put one focused workflow per file unless existing patterns justify more.
2. Use installed command names like `simtools-validate-optics`.
3. Keep `test_name` stable, short, and unique for the application.
4. Keep generated path settings such as `output_path`, `grid_output_path`, and
   `pack_for_grid_register` relative; the harness rewrites them into a
   temporary test directory.
5. Add `test: true`, small event counts, `n_workers: 1`, or short ranges when
   the application supports them.

## Application-Level Options

Optional application-level keys beside `application` and `configuration`:

- `model_version_use_current: true`: run only when the CLI `--model_version`
  matches the config model version.
- `skip_integration_test: <reason>`: temporary explicit skip with reason.
- `test_use_case: UC-...`: add use-case pytest marker.
- `test_requirement: REQ-...`: add requirement pytest marker.
- `xfail_network_error: true`: xfail only recognized network failures.
- `docs.title` / `docs.summary`: metadata for generated documentation
  examples.
- `preparation`: schema `0.6.0` steps using `simtools-get-model-parameter` to
  create temporary inputs. Reference these with `${prepared:path/to/file}` in
  the application configuration.

Use `configuration.<option>.by_version` for version-dependent CLI values:

```yaml
array_layout_name:
  by_version:
    "<7.0.0": alpha
    ">=7.0.0": CTAO-South-Alpha
```

## Test Resources

Use `${static:path/to/file}` for maintained resources,
`${generated:path/to/file}` for generated resources, and
`${downloaded:path/to/file}` for externally downloaded resources. Pytest
resolves these against the versioned `simtools-tests` resource bundle selected
by `--simtools_tests_path` and `--simtools_tests_resource_version`, or their
`SIMTOOLS_TESTS_PATH` and `SIMTOOLS_TESTS_RESOURCE_VERSION` environment
variables. The default resource version comes from `dependency_versions.yml`.
Keep maintained resources in the versioned bundle rather than using raw
`tests/resources/...` paths.

## `integration_tests` Blocks

Declare artifacts with `test_outputs`; each item owns its location and an
optional ordered list of explicit `validations`. Use the strongest cheap
validation available:

```yaml
integration_tests:
  - test_outputs:
    - file: results/output.ecsv
      path_descriptor: output_path
      validations:
      - type: table
        minimum_rows: 1
    - file: run.simtel.zst
      path_descriptor: pack_for_grid_register
      validations:
      - type: simtel
        event_type: shower
        event:
          pe_sum: {range: [20, 1000]}
          trigger_time: {range: [0, 50]}
```

Validation keys:

- `test_outputs`: a list of generated artifacts. Each requires `file` and
  `path_descriptor`; `output_sub_path` is optional.
- `validations`: explicit validators. Available types are `format`,
  `reference`, `data_schema`, `table`, `metadata`, `hdf5_datasets`,
  `hdf5_product`, `log`, `simtel`, `simtel_config`, and `model_parameter`.
- `reference`: compare JSON, YAML, or ECSV references. ECSV comparison can
  select `columns`, specify `key_columns`, include `metadata`, and filter rows.
- `simtel`: validate event type, event ranges, and sim_telarray metadata.
  `metadata` checks required keys and relations. `simtel_config` compares
  generated sim_telarray configuration files.
- `log`: validate expected and forbidden log patterns.

## Commands

```bash
pytest --no-cov tests/integration_tests/test_applications_from_config.py
pytest -v -k "simtools-<app-name>" tests/integration_tests/test_applications_from_config.py
pytest -v -k "simtools-<app-name>_<test_name>" \
  tests/integration_tests/test_applications_from_config.py
pytest -v --model_version 6.0.2 -k "<test_name>" \
  tests/integration_tests/test_applications_from_config.py
pytest -v --simtools_tests_path /full/path/to/simtools-tests/simtools-tests \
  --simtools_tests_resource_version <version> \
  tests/integration_tests/test_applications_from_config.py
```

## Debug Checklist

1. Confirm the selected filesystem or Git model source and model version are
   available.
2. Confirm expected files use the post-rewrite temp paths via `output_path` or
   `pack_for_grid_register`.
3. Prefer filename existence checks first, then add reference or physics-range
   checks for critical products.
4. If a version matrix fails, check `model_version`, `by_version`,
   `model_version_use_current`, and version-specific expected filenames.
5. Keep generated files deterministic by fixing seeds, run numbers, labels,
   event counts, and worker counts.
6. If sim_telarray files are missing, inspect the application log path printed in
   stderr before changing filenames. For log-validation failures, inspect stdout
   and stderr for actual errors, exceptions, tracebacks, warnings, or crashes.
