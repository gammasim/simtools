---
name: integration-testing
description: >-
  Create, update, or debug simtools integration test YAML configs for
  application workflows in tests/integration_tests/config, including MongoDB
  prerequisites, model-version handling, and output validation blocks.
---

# Integration Testing for simtools

Use this skill for integration test configuration files under
`tests/integration_tests/config/`. These tests run full `simtools-*`
applications from YAML and validate their outputs.

Follow `.github/copilot-instructions.md` and
`docs/source/developer-guide/testing.md`. For exact mechanics, inspect
`tests/integration_tests/test_applications_from_config.py`,
`tests/integration_tests/conftest.py`, and `src/simtools/testing/`.

## Prerequisite

Integration tests require a valid MongoDB connection via `.env` settings.
For local container setup, use `docker/README.md`; when using a local database,
run the dev container on `simtools-mongo-network`.

## Config Shape

```yaml
---
applications:
- application: simtools-<app-name>
  configuration:
    model_version: 6.0.2
    output_path: simtools-output
    # application CLI options as YAML keys
  integration_tests:
  - output_file: relative/file/from/output_path.ext
  test_name: short_descriptive_case
schema_name: application_workflow.metaschema
schema_version: 0.4.0
```

Rules:

1. Put one focused workflow per file unless existing patterns justify more.
2. Use installed command names like `simtools-validate-optics`.
3. Keep `test_name` stable, short, and unique for the application.
4. Keep `output_path` and `pack_for_grid_register` relative; the harness
   rewrites them into a temporary test directory.
5. Use realistic CTAO names and conventions: `North`/`South`, `LSTN-01`,
   `MSTS-05`, semantic model versions without a leading `v`.
6. Add `test: true`, small event counts, `n_workers: 1`, or short ranges when
   the application supports them.

## Application-Level Options

Optional keys beside `application`, `configuration`, `integration_tests`, and
`test_name`:

- `model_version_use_current: true`: run only when the CLI `--model_version`
  matches the config model version.
- `skip_for_production_db: true`: skip DB-writing tests on production DBs.
- `skip_integration_test: <reason>`: temporary explicit skip with reason.
- `test_use_case: UC-...`: add use-case pytest marker.
- `test_requirement: REQ-...`: add requirement pytest marker.
- `xfail_network_error: true`: xfail only recognized network failures.

Use `configuration.<option>.by_version` for version-dependent CLI values:

```yaml
array_layout_name:
  by_version:
    "<7.0.0": alpha
    ">=7.0.0": CTAO-South-Alpha
```

## `integration_tests` Blocks

Use the strongest cheap validation available:

```yaml
integration_tests:
- output_file: results/output.ecsv
- file_type: json
- test_output_files:
  - file: run.log
    path_descriptor: output_path
    output_sub_path: sim_telarray/run000010
  - file: run.simtel.zst
    path_descriptor: pack_for_grid_register
    expected_sim_telarray_output:
      pe_sum: [20, 1000]
      photons: [90, 1000]
      trigger_time: [0, 50]
- reference_output_file: tests/resources/reference.ecsv
  test_output_file: results/output.ecsv
  tolerance: 1.e-2
- model_parameter_validation:
    parameter_file: nsb_pixel_rate/nsb_pixel_rate-0.0.99.json
    reference_parameter_name: nsb_pixel_rate
    tolerance: 1.e-1
    scaling: 10.0
- test_simtel_cfg_files:
    "6.0.2": tests/resources/sim_telarray_configurations/6.0.2/CTA-South-LSTS-01_test.cfg
```

Validation keys:

- `output_file`: file under `configuration.output_path`.
- `test_output_files`: list or single mapping with `file`, `path_descriptor`,
  optional `output_sub_path`, and optional sim_telarray/log expectations.
- `file_type`: checks JSON/YAML parsing; other types check suffix only.
- `reference_output_file`: compare ECSV, JSON, YAML, or YML; optional
  `test_output_file`, `tolerance`, and ECSV `test_columns`.
- `model_parameter_validation`: compare generated parameter JSON to MongoDB.
- `test_simtel_cfg_files`: compare generated sim_telarray cfg for matching
  model version.

For log files, use `expected_log_output.pattern` and `forbidden_pattern`.
For sim_telarray files, use `expected_sim_telarray_output` and
`expected_sim_telarray_metadata`.

## Commands

```bash
pytest --no-cov tests/integration_tests/test_applications_from_config.py
pytest -v -k "simtools-<app-name>" tests/integration_tests/test_applications_from_config.py
pytest -v -k "simtools-<app-name>_<test_name>" \
  tests/integration_tests/test_applications_from_config.py
pytest -v --model_version 6.0.2 -k "<test_name>" \
  tests/integration_tests/test_applications_from_config.py
```

## Debug Checklist

1. Confirm `.env` contains MongoDB credentials and model version settings.
2. Confirm expected files use the post-rewrite temp paths via `output_path` or
   `pack_for_grid_register`.
3. Prefer filename existence checks first, then add reference or physics-range
   checks for critical products.
4. If a version matrix fails, check `model_version`, `by_version`,
   `model_version_use_current`, and version-specific expected filenames.
5. Keep generated files deterministic by fixing seeds, run numbers, labels,
   event counts, and worker counts.
