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

### Run with simulation models from files

To run the integration tests without MongoDB, pass the directory containing
`simulation-models/productions` and `simulation-models/model_parameters`:

```bash
pytest --no-cov -v \
  --model_version 7.0.0 \
  --simulation_models_path ../simulation-models \
  tests/integration_tests/test_applications_from_config.py
```

The integration harness passes the path to each application subprocess. To run one workflow:

```bash
pytest --no-cov -vv \
  --model_version 7.0.0 \
  --simulation_models_path ../simulation-models \
  'tests/integration_tests/test_applications_from_config.py::test_applications_from_config[simtools-docs-produce-array-element-report_run]'
```

The filesystem source takes precedence over MongoDB settings. Integration workflows marked with
`requires_mongodb: true` are skipped.

## Resources

Tests resolve resources from the path in `SIMTOOLS_TEST_RESOURCES`. If no full
resource path is configured, `SIMTOOLS_TESTS_PATH` identifies the
`simtools-tests` checkout. The default tag is maintained in
`dependency_versions.yml`; the command-line option `--simtools_tests_tag` can
select a different tag for an individual run. `SIMTOOLS_TESTS_TAG` is the
canonical environment override; `SIMTOOLS_TESTS_VERSION` remains supported as
an alias.

```text
<simtools-tests>/simtools-tests/<selected-version>/integration_tests/
  static/
  generated/
  downloaded/
```

Use `${static:path/to/file}` for maintained inputs and
`${generated:path/to/file}` for generated reference products.
Use `${downloaded:path/to/file}` for externally downloaded resources.
To run against a different resource set:

```bash
pytest --test_resources_path /full/path/to/resources \
  tests/integration_tests/test_applications_from_config.py
```

To select a version instead of a path:

```bash
pytest --simtools_tests_tag v0.36.0 \
  tests/integration_tests/test_applications_from_config.py
```

When no version is specified, simtools uses the version in `dependency_versions.yml`.

Versioned resource bundles are archived in
[`gammasim/simtools-tests`](https://github.com/gammasim/simtools-tests). Use
the resource applications to create and synchronize these bundles:

- [`simtools-resources-test-generate`](../user-guide/applications/simtools-resources-test-generate.md)
  generates versioned test resources and validates configured static files.
The resource generation and release workflow is documented in
[Test resources](testing_resources.md).

Use the current versioned resource set from `simtools-tests` for development,
PR CI, and compatibility checks.

## Validation

Declare generated artifacts in `test_outputs`. Every declared artifact must
exist. Add an ordered `validations` list when content also needs validation.
Validator selection is explicit and does not depend on the filename suffix.

Available validator types are `format`, `reference`, `data_schema`, `table`,
`metadata`, `hdf5_datasets`, `hdf5_product`, `log`, `simtel`,
`simtel_config`, and `model_parameter`.

The validation interface has no legacy aliases: output checks must use
`test_outputs` and explicit validator types.

Keep generated outputs deterministic by fixing seeds, labels, event counts,
worker counts, and version-specific expectations.

## Composable output validation

Each output owns its location and validation rules. Product schemas validate
stable structure such as columns, types, and units. Table and metadata rules
describe expectations specific to the tested workflow.

```yaml
test_outputs:
- file: job_grid.ecsv
  path_descriptor: output_path
  validations:
  - type: data_schema
    schema: src/simtools/schemas/job_grid_density.schema.yml
  - type: table
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
  - type: metadata
    required_keys: [job_grid_summary]
    relations:
    - left: job_grid_summary.simulation_rows
      equals: table.row_count
    - left: job_grid_summary.total_showers
      equals: table.column_sum
      column: showers_per_run
```

`minimum_rows` rejects empty or unexpectedly short tables. `unique_columns`
checks complete columns for duplicate values. Column rules support
`allowed_values` and inclusive or exclusive numerical `range` bounds, with an
optional unit. Metadata paths use dotted mapping notation.

Metadata relations compare a dotted metadata path with either the table row
count or a named column sum.

Table and metadata validators use Astropy table format auto-detection, so they
can validate any table format supported by the installed Astropy I/O registry.

The `simtel` validator defaults to shower events and accepts explicit event
ranges without exposing the underlying reader API:

```yaml
- type: simtel
  event_type: shower
  event:
    pe_sum: {range: [20, 1000]}
    trigger_time: {range: [0, 50]}
```

The `reference` validator compares JSON, YAML, or ECSV files. ECSV comparisons
check row order, column values and units by default. Use `columns` to select a
subset, `key_columns` to compare rows in deterministic key order, `metadata`
to include table metadata, and typed `filters` with operators such as `equal`,
`less`, `greater_equal`, `in`, or `not_in`. Reference filtering never executes
configuration text as code.
