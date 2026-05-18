---
name: unit-testing
description: "Add or improve simtools unit tests by using the repo conventions plus an annotated coverage workflow."
---

# Unit Testing for simtools

Use this skill for unit tests in `tests/unit_tests/` and for coverage work on library code in `src/simtools/`.

Follow `.github/copilot-instructions.md` for the repository rules. This file adds the unit-test workflow, with coverage guidance first.

## Coverage first

Coverage is the main driver for this skill. For changed library code, aim for full coverage of the changed lines and branches.

Run focused annotated coverage first:

```bash
pytest tests/unit_tests/path/to/test_module.py \
  --cov=simtools.path.to.module --cov-report=annotate:cov_annotate
```

If the change affects shared behavior, run broader coverage:

```bash
pytest --cov=simtools --cov-report=annotate:cov_annotate
```

Open the matching file in `cov_annotate/`. Lines starting with `!` are not covered and should drive the next tests you add.

## Workflow

1. Find the production module in `src/simtools/` and the mirrored test file in `tests/unit_tests/`.
2. Reuse existing local patterns and fixtures before adding new helpers.
3. Add small tests that cover the missing lines and the relevant success, error, and branch paths.
4. Re-run the focused test, then run broader unit tests only if the change affects shared behavior.

## Test rules

1. Mirror `src/simtools/` under `tests/unit_tests/`.
2. Use plain pytest test functions, not test classes.
3. Cover every changed function or method, including success paths, error paths, and branches.
4. Reuse fixtures from `tests/unit_tests/conftest.py` before adding new helpers.
5. Use `tmp_test_directory` for file I/O. Do not use `tmp_path`, `/tmp`, or hardcoded absolute temp paths.
6. Mock external dependencies such as databases, network calls, file I/O, and installed simulation software.
7. Use `pytest.approx()` for floats and `astropy.tests.helper.assert_quantity_allclose` for quantities.
8. Keep tests fast and readable.

## Useful Repository Notes

- Shared fixtures in `tests/unit_tests/conftest.py` already mock simulator paths for most unit tests.
- Relevant markers:
  - `uses_model_database`
  - `db_unit_test`

## Commands

```bash
pytest
pytest -vv tests/unit_tests/model/test_foo.py::test_bar
pytest --durations=10 tests/unit_tests/
pytest --lf tests/unit_tests/
pytest --random-order tests/unit_tests/
pytest -n 4 --dist loadscope
pre-commit run --all-files
```

CI runs unit tests with:

```bash
pytest --durations=10 --color=yes -n 4 --dist loadscope \
  --cov=simtools --cov-report=xml --retries 2 --retry-delay 5
```

## Setup

```bash
pip install -e '.[tests]'
pip install -e '.[dev,tests]'
```
