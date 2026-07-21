# Integration Tests

Integration tests run installed `simtools-*` applications from version-controlled
workflow files in `tests/integration_tests/config/` and validate their outputs.

## Scope

Integration tests should cover:

- representative command-line use cases for each application
- external interfaces such as the model database and downloaded input files
- internal interfaces where one `simtools` output becomes another input
- selected compatibility checks for generated products

Keep pull-request integration tests within typical GitHub CI runtime. Prefer
cheap checks first and stronger numerical or file-based validation where it
adds signal.

## Workflow Files

Each workflow file should define:

- `application`
- `configuration`
- `docs.title` and `docs.summary`
- `test_name`
- `integration_tests`
- optional `test_use_case` and `test_requirement`

One focused workflow per file is preferred unless an existing application
pattern already groups related cases.

## Commands

Run all integration tests:

```bash
pytest --no-cov tests/integration_tests/test_applications_from_config.py
```

Run one application:

```bash
pytest -v -k "simtools-<app-name>" \
  tests/integration_tests/test_applications_from_config.py
```

Run one workflow:

```bash
pytest -v -k "simtools-<app-name>_<test_name>" \
  tests/integration_tests/test_applications_from_config.py
```

Run with a selected model version:

```bash
pytest -v --model_version 6.0.2 -k "<test_name>" \
  tests/integration_tests/test_applications_from_config.py
```

## Resources

By default, tests resolve resources from `tests/resources`:

```text
tests/resources/
  static/
  generated/
  downloaded/
```

Use `${static:path/to/file}` for maintained inputs and
`${generated:path/to/file}` for generated reference products.
Use `${downloaded:path/to/file}` for externally downloaded resources.
To run against a different resource set:

```bash
pytest --test-resources-path /full/path/to/resources \
  tests/integration_tests/test_applications_from_config.py
```

Versioned resource bundles are archived in
[`gammasim/simtools-tests`](https://github.com/gammasim/simtools-tests). Use
the resource applications to create and synchronize these bundles:

- [`simtools-resources-test-generate`](../user-guide/applications/simtools-resources-test-generate.md)
  generates versioned test resources and validates configured static files.
- [`simtools-resources-test-sync`](../user-guide/applications/simtools-resources-test-sync.md)
  compares a versioned bundle against `tests/resources` and optionally syncs it.

The resource generation and release workflow is documented in
[Test resources](testing_resources.md).

Use `tests/resources` for normal development and PR CI. Use archived resource
sets from `simtools-tests` explicitly when validating compatibility against a
named release.

## Validation

Prefer explicit validation blocks over exit-code-only tests. Common patterns:

- `output_file` for required outputs
- `file_type` for parseable JSON or YAML outputs
- `reference_output_file` with `tolerance` for numerical regression checks
- `model_parameter_validation` for DB-backed parameter checks
- `test_output_files` with `expected_log_output`,
  `expected_sim_telarray_output`, or `expected_simtel_metadata`
- `test_simtel_cfg_files` for version-dependent sim_telarray configuration

Keep generated outputs deterministic by fixing seeds, labels, event counts,
worker counts, and version-specific expectations.
