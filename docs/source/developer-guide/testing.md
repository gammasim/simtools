# Testing

Automatic code testing is absolutely essential for the development of simtools. It ensures that the code is working as expected and that new features do not break existing functionality.

```{note}
Developers should expect that code changes affecting several modules are acceptable in case unit tests are successful.
```

The testing should be done at two levels:

- unit tests: test individual modules and functions
- integration tests: test applications as a whole using typical use cases

The [pytest](https://docs.pytest.org) framework is used for testing.
The test modules are located in
[./tests](https://github.com/gammasim/simtools/tree/main/tests) separated
by unit and integration tests.

:::{hint}
The rule should be that any discovered bug or issue should trigger the implementation of tests which reproduce the issue and prevent it from reoccurring.
:::

## Unit tests

```{important}
**Every function and module in the `simtools` library code must have at least one test.**
No library function should be left untested — **tests are mandatory, not optional.** (Application code is excluded from this requirement; see the developer guidelines for details.)
```

Unit tests should verify the behavior of individual functions ("units").
Write tests in parallel with code development to ensure modules are testable from the start.

```{note}
The `simtools` project aims for **very high test coverage**.
Uncovered lines must be the rare exception. The project mandates at least 90% coverage as a minimum threshold, but developers are encouraged to aim for coverage as close to 100% as possible.
```

Check the test coverage with `pytest -n auto --cov-report-html  tests/unit_tests/ tests/integration_tests/`.
Coverage reports can be accessed using a browser through the `htmlcov/index.html` file.

Good practice:

- at least one test for every function in a module clearly testing the function's behavior (including edge cases).
- sort tests in the testing module in the same sequence as the functions in the module.
- unit tests need to be fast. Add `--durations=10` to pytest to see slow tests.
- use mocking to avoid external dependencies (e.g., database connections, file I/O); see below
- use the fixture `tmp_test_directory` to create a temporary test directory for file I/O tests. Do not use `tmp_path` or the `tempfile` module.

### Pytest fixtures

Shared utilities for tests (e.g., DB connections) are provided as `pytest.fixtures` in
[conftest.py](https://github.com/gammasim/simtools/blob/main/tests/conftest.py). Use these to avoid duplication.

- Define fixtures as locally as possible:
  - If used only in a single test module, place them at the top of that module.
  - If reused across modules, put them in `conftest.py`.

- List all available fixtures with:

```bash
pytest --fixtures
```

- Fixtures defined in [conftest.py](https://github.com/gammasim/simtools/blob/main/tests/conftest.py) must have a docstring;
- Local fixtures should include a docstring if it aids readability.

### Mocking

Mocking is used to replace parts of the system under test and make assertions about how they have been used.
Mocking is used to avoid external dependencies (e.g., database connections, file I/O).

The [unittest.mock](https://docs.python.org/3/library/unittest.mock.html) module is used for mocking.
The `mock` module is used to replace parts of the system under test and make assertions about how they have been used. An example is e.g.

```python
    with mock.patch(
        "simtools.simtel.simtel_table_reader._read_simtel_data_for_atmospheric_transmission"
    ) as mock_read:
        simtel_table_reader.read_simtel_table("atmospheric_transmission", "test_file")
        mock_read.assert_called_once()
```

### Assertion of floats and astropy quantities

When asserting floating point values, use `pytest.approx()` to avoid issues with precision:

```python
def test_floating_point_comparison():
    assert 0.1 + 0.2 == pytest.approx(0.3)
```

`pytest.approx(expected, rel=None, abs=None, nan_ok=False)`.

Use `astropy.tests.helper` for testing astropy quantities:

```python
from astropy.tests.helper import assert_quantity_allclose

def test_astropy_quantity_comparison():
    assert_quantity_allclose(0.1 * u.m, 0.1 * u.m)
```

## Integration tests

Integration tests should be written for every application and cover the most important use cases.
Integration tests should follow the logic of the existing tests in [simtools/tests/integration_tests](https://github.com/gammasim/simtools/tree/main/tests/integration_tests/).
Any configuration file in the [simtools/tests/integration_tests/config](https://github.com/gammasim/simtools/tree/main/tests/integration_tests/config) directory is used for testing.

For testing a single applications, use the `-k` option to select the test by name:

```bash
pytest -v -k "simtools-convert-all-model-parameters-from-simtel" tests/integration_tests/test_applications_from_config.py
```

To run a specific tests, use the `-k` option with the test name:

```bash
pytest -v -k "simtools-convert-all-model-parameters-from-simtel_num_gains" tests/integration_tests/test_applications_from_config.py
```

This runs to run the tool for the specific test called `num_gains`.

### Validation of test outputs

The integration test module allows to compare test outputs with expected outputs or files (see the [simtools/tests/integration_tests/config](https://github.com/gammasim/simtools/tree/main/tests/integration_tests/config) directory for examples).

#### Test that a given output file exists

Test that a given output file is created by the tool in the simtools output directory:

```text
integration_tests:
- output_file: ray-tracing/results/ray-tracing-North-LSTN-01-d11.0km-za20.0deg_validate_optics.ecsv
```

Test that a given output file is created by the tool in the provided output directory:

```text
integration_tests:
- test_output_files:
  - file: proton_run000001_za20deg_azm180deg_North_alpha_6.0.2_check_output.log_hist.tar.gz
      path_descriptor: pack_for_grid_register
```

Test that output file is of a given type (e.g. json).
Test is successful for certain equivalent file types (e.g. yml vs yaml).

```text
- file_type: json
```

#### Compare with reference file

Compare tool output with a given reference files (in the following example with a tolerance of 1.e-2 for floating point comparison):

```text
integration_tests:
  - reference_output_file: ray-tracing/results/ray-tracing-North-LSTN-01-d11.0km-za20.0deg_validate_optics.ecsv
    tolerance: 1.e-2
    test_output_file: ray_tracing_results.ecsv
```

If no `test_output_file` is given, the output file defined in the configuration file is used.
Tests are implemented for any astropy table-like files.

The tests can be restricted to certain rows.
The following examples compares only rows with `best_fit==True` and only the column
`mirror_reflection_random_angle_sigma1` with a tolerance of 0.3:

```text
integration_tests:
  - reference_output_file: tests/resources/derive_mirror_rnda_psf_random_flen.ecsv
    test_columns:
    - cut_column_name: best_fit
      cut_condition: ==True
      test_column_name: mirror_reflection_random_angle_sigma1
    tolerance: 0.3
```

#### Compare with reference values from model parameter DB

Test outputs of derived model parameters can be compared with reference values stored in the model parameter database.

```text
integration_tests
  - model_parameter_validation:
      parameter_file: nsb_pixel_rate/nsb_pixel_rate-0.0.99.json
      reference_parameter_name: nsb_pixel_rate
      tolerance: 1.e-1
```

Some tests might return a scaled value of the model parameter as defined for the given model. In this case, the scaling factor can be defined as well:

```text
integration_tests
  - model_parameter_validation:
      parameter_file: nsb_pixel_rate/nsb_pixel_rate-0.0.99.json
      reference_parameter_name: nsb_pixel_rate
      tolerance: 1.e-1
      scaling: 10.0  # camera efficiency changed by factor 0.1
```

#### Test simulation output

Test event data generated generated by sim_telarray has certain characteristics.
The test compares e.g., that the mean number of photons generated for all events
in the test is within the expected range.

```text
integration_tests:
- expected_output:
        pe_sum:
        - 20
        - 1000
        photons:
        - 90
        - 1000
        trigger_time:
        - 0
        - 50
      file: proton_run000001_za20deg_azm180deg_North_alpha_6.0.2_check_output.simtel.zst
      path_descriptor: pack_for_grid_register
```

Test the sim_telarray metadata:

```text
integration_tests:
    - expected_simtel_metadata:
        site_config_name: South
        1:
          effective_focal_length: '2923.7 0 0 2 0'
        38:
          effective_focal_length: '215.191 0 0 0 0'
      file: gamma_run000020_za62deg_azm180deg_South_beta_6.0.2_test.simtel.zst
      path_descriptor: pack_for_grid_register
```

Test for required and forbidden patterns in log files:

```text
integration_tests:
    - expected_log_output:
        pattern:
        - "END OF SHOWER NO          5"
        - "========== END OF RUN ========"
        - "Sim_telarray finished at"
        - "CURVED VERSION WITH SLIDING PLANAR ATMOSPHERE"
        - "CORSIKA was compiled with CURVED option."
      file: gamma_run000020_za62deg_azm180deg_South_beta_6.0.2_test.log_hist.tar.gz
      path_descriptor: pack_for_grid_register
      forbidden_pattern:
      - "Error"
```

Test sim_telarray configuration files against reference files for different model versions:

```text
integration_tests:
  - test_simtel_cfg_files:
      "5.0.0": tests/resources/sim_telarray_configurations/5.0.0/CTA-South-LSTS-01_test.cfg
      "6.0.2": tests/resources/sim_telarray_configurations/6.0.2/CTA-South-LSTS-01_test.cfg
  - test_simtel_cfg_files:
      "5.0.0": tests/resources/sim_telarray_configurations/5.0.0/CTA-South-MSTS-01_test.cfg
      "6.0.2": tests/resources/sim_telarray_configurations/6.0.2/CTA-South-MSTS-01_test.cfg
```

### Model versions for integration tests

The integration tests run for two different model versions, defined in the matrix in the [CI-integrationtests.yml](https://github.com/gammasim/simtools/blob/main/.github/workflows/CI-integrationtests.yml) workflow.

As model versions and definitions change, not all integration tests will work with all model versions.
Use the entry `MODEL_VERSION_USE_CURRENT` in the configuration file to fix the model version to the one defined in the configuration file.

## Testing utilities

The [pytest-xdist](https://pytest-xdist.readthedocs.io/en/latest/) plugin is part of the developer environment
and can be used to run unit and integration tests in parallel (e.g., `pytest -n 4` to run on four cores in parallel).

Tests might pass just because they run after an unrelated test. In order to test the independence of unit tests, use the
[pytest-random-order](https://pypi.org/project/pytest-random-order/) plugin with `pytest --random-order`.

## Profiling tests

Tests should be reasonably fast.

To identify slow tests, run the tests with the `--durations` option to list e.g. the 10 slowest tests:

```bash
pytest --durations=10
```

For profiling, use the [pytest-profiling](https://pypi.org/project/pytest-profiling/) plugin.
Example:

```bash
pytest --no-cov --profile tests/unit_tests/utils/test_general.py
```

To generate flame graphs for the profiling results:

```bash
pytest --no-cov --profile-svg tests/unit_tests/utils/test_general.py
```

(the installation of [graphviz](https://graphviz.org/) is required to generate the graphs)
