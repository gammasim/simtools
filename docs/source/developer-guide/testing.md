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
**Every function and module in `simtools` must have at least one test.**
No function should be left untested — **tests are mandatory, not optional.**
```

Unit tests should verify the behavior of individual functions ("units").
Write tests in parallel with code development to ensure modules are testable from the start.

```{note}
The `simtools` project aims for **very high test coverage**.
Uncovered lines must be the rare exception. The goal is coverage close to 100%.
(simtools quality requirements mandate at least 90% coverage.)
```

Check the test coverage with `pytest -n auto --cov-report-html  tests/unit_tests/ tests/integration_tests/`.
Coverage reports can be accessed using a browser through the `htmlcov/index.html` file.

Good practice:

- at least one test for every function in a module clearly testing the function's behavior (including edge cases).
- sort tests in the testing module in the same sequence as the functions in the module.
- unit tests need to be fast. Add `--durations=10` to pytest to see slow tests.
- use mocking to avoid external dependencies (e.g., database connections, file I/O); see below

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

Add appropriate tolerances, if necessary:
`pytest.approx(expected, rel=None, abs=None, nan_ok=False)[source]`.

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

The integration test module allows for additional output tests (see the [simtools/tests/integration_tests/config](https://github.com/gammasim/simtools/tree/main/tests/integration_tests/config) directory for examples).

Test that a given output file exists:

```text
  INTEGRATION_TESTS:
    - OUTPUT_FILE: ray-tracing/results/ray-tracing-North-LSTN-01-d11.0km-za20.0deg_validate_optics.ecsv
```

Compare tool output with a given reference files (in the following example with a tolerance of 1.e-2 for floating point comparison):

```text
REFERENCE_OUTPUT_FILES:
  - REFERENCE_FILE: ray-tracing/results/ray-tracing-North-LSTN-01-d11.0km-za20.0deg_validate_optics.ecsv
    TOLERANCE: 1.e-2
```

This is implemented for any astropy table-like file.

Test for a certain file type:

```text
- FILE_TYPE: json
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
