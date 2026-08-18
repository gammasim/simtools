# Unit Tests

Unit tests cover library code in `src/simtools/` at module and function level.
Application entrypoints in `src/simtools/applications/` are covered by
integration tests instead.

## Scope

- Add at least one focused test for every changed library function or method.
- Cover success paths, error paths, and important branches.
- Keep tests fast and deterministic.
- Mirror the package structure under `tests/unit_tests/`.

The project target is at least 90% coverage. New code should stay close to
full branch and line coverage.

## Commands

Run all unit tests:

```bash
pytest tests/unit_tests/
```

Run focused coverage for one module:

```bash
pytest tests/unit_tests/path/to/test_module.py \
  --cov=simtools.path.to.module --cov-report=annotate:cov_annotate
```

Useful options:

```bash
pytest --durations=10 tests/unit_tests/
pytest --random-order tests/unit_tests/
pytest -n 4 --dist loadscope tests/unit_tests/
```

## Test Design

- Reuse fixtures from `tests/unit_tests/conftest.py` and `tests/conftest.py`
  before adding new helpers.
- Keep mutable settings, database clients, simulator paths, and I/O handlers
  explicit in the test or module that needs them. Global autouse fixtures are
  reserved for harmless process-wide safety such as the matplotlib backend.
- Prefer one parametrized test with descriptive case IDs for an invariant that
  has several inputs. Keep distinct error contracts and boundary branches as
  separate cases in the same table.
- Every collected test must have an explicit oracle: an assertion, an
  `assert_*` helper, a `pytest.raises`/`pytest.warns` context, or a mock
  assertion. Assertions such as `is not None`, `.called`, or file existence
  alone are not sufficient when the returned value or file content is known.
- Do not make unit tests depend on checked-in, downloaded, or external files.
  When file parsing or writing is the behavior under test, generate the
  smallest valid input in `tmp_test_directory` within the test.
- Keep file-format compatibility and resource-heavy checks in integration tests.
- Mock external dependencies such as databases, network access, file downloads,
  and installed simulation software.
- Use `pytest.approx()` for floating-point values.
- Use `astropy.tests.helper.assert_quantity_allclose` for astropy quantities.

Keep local fixtures close to the test module that uses them.

## Coverage Workflow

Coverage is a diagnostic tool, not the goal by itself. Use it to find missing
branches after the semantic behavior is covered.

```bash
pytest --cov-branch --cov-report=html --cov-report=json tests/unit_tests/
```

Open the file `htmlcov/index.html` in your browser to see the coverage report.
