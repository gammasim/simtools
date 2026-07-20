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

An integration test may include an optional `output_validation` list. Each file
rule declares `kind`, `path_descriptor`, and `file`. Supported kinds are
`file`, `table`, `mapping`, `group`, and `event_stream`.

```yaml
output_validation:
- name: job_grid_content
  kind: table
  path_descriptor: output_path
  file: job_grid.ecsv
  format: ecsv
  non_empty: true
  count:
    minimum: 1
  required_columns: [run_number, model_version]
  columns:
    run_number:
      type: int64
      finite: true
      unique: true
    energy_min:
      type: float64
      range:
        minimum: 30.0
        maximum: 300.0
        unit: GeV
  metadata:
    required_keys: [job_grid_summary]
  consistency:
  - left:
      source: metadata
      path: job_grid_summary.simulation_rows
    operator: equals
    right:
      source: content
      metric: row_count
  data_product_schema: src/simtools/schemas/job_grid_density.schema.yml
```

`non_empty` checks bytes for a generic file, rows for a table, keys for a
mapping or HDF5 group, and events for an event stream. `count` supports
`exact`, `minimum`, and `maximum`. Column rules support declared data types,
units, finite values, uniqueness, allowed values, and inclusive numerical
ranges. Required and forbidden keys or columns are checked before field rules.

When `data_product_schema` is provided, schema-owned constraints such as
required columns, column types, and units can be omitted from the validation
rule. Keep the rule focused on content invariants and workflow-specific domains
that are not part of the general product schema.

Consistency operands use only structured references: metadata paths such as
`job_grid_summary.total_showers`, content metrics such as `row_count`, and
column aggregates (`sum`, `minimum`, `maximum`, `count`, or `unique_count`).
Supported operators are `equals`, `greater_than`, `greater_equal`,
`less_than`, `less_equal`, and `in`.

Use a `kind: relationship` rule only when the values being compared are in
different output files. Each entry in `outputs` gives a generated file a local
name. A check operand selects that name with `output`, then reads either a
dotted mapping or metadata `path`, or a content `metric` such as `row_count`.
For example, if an application writes `jobs.ecsv` and a separate
`summary.yml`, this checks that the summary reports the actual number of rows:

```yaml
output_validation:
- name: summary_matches_jobs
  kind: relationship
  outputs:
  - name: jobs
    kind: table
    path_descriptor: output_path
    file: jobs.ecsv
    format: ecsv
  - name: summary
    kind: mapping
    path_descriptor: output_path
    file: summary.yml
  checks:
  - left:
      output: summary
      path: simulation_rows
    operator: equals
    right:
      output: jobs
      metric: row_count
```

If `summary.yml` reports `3` while `jobs.ecsv` has `2` rows, the relationship
fails. If the summary is metadata inside `jobs.ecsv`, use a `consistency` rule
instead; that compares metadata and content within one output.
