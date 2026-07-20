# Integration Tests

Integration tests run installed `simtools-*` applications from version-controlled
workflow files in `tests/integration_tests/config/` and validate their outputs.

## Scope

Integration tests should cover:

- representative command-line use cases for each application
- external interfaces such as the model database and downloaded input files
- internal interfaces where one `simtools` output becomes another input
- selected compatibility checks for generated products

The tests follow the following levels of assertion, from weakest to strongest:

- execution only;
- file presence;
- parse or schema validation;
- semantic invariants;
- deterministic reference comparison.

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

- [`simtools-resources-test-generate`](../user-guide/applications/simtools-resources-test-generate.rst)
  generates versioned test resources and validates configured static files.
- [`simtools-resources-test-sync`](../user-guide/applications/simtools-resources-test-sync.rst)
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

## Declarative output validation

An integration test may include an optional `output_validation` list for
generated ECSV tables. Each rule declares the output location and its
data-product schema. The schema validates required columns, types, units, and
finite numerical values; keep the workflow rule focused on invariants specific
to the integration-test configuration.

```yaml
output_validation:
- name: job_grid_content
  path_descriptor: output_path
  file: job_grid.ecsv
  data_product_schema: src/simtools/schemas/job_grid_density.schema.yml
  minimum_rows: 1
  unique_columns: [run_number]
  columns:
    primary:
      allowed_values: [gamma, proton]
    energy_min:
      range:
        minimum: 30.0
        maximum: 300.0
        unit: GeV
  metadata:
    required_keys: [job_grid_summary]
    row_count: job_grid_summary.simulation_rows
    column_sums:
      showers_per_run: job_grid_summary.total_showers
```

`minimum_rows` rejects empty or unexpectedly short tables. `unique_columns`
checks complete columns for duplicate values. Column rules support
`allowed_values` and inclusive or exclusive numerical `range` bounds, with an
optional unit. Metadata paths use dotted mapping notation.

`metadata.row_count` names a metadata value that must equal the number of table
rows. Each `metadata.column_sums` entry maps a table column to metadata that
must equal that column's sum.
