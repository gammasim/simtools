# simtools Agent Guide

This file gives repo-wide instructions for AI agents working on simtools.

simtools is a Python toolkit for CTAO Monte Carlo production support: model
parameter handling, MongoDB access, CORSIKA and sim_telarray configuration,
application workflows, validation, reporting, and plotting.

## First Steps

1. Inspect the existing code and tests before changing behavior.
2. Prefer the narrowest change that fixes the requested issue.
3. Check whether a local skill applies:
   - `.agents/skills/unit-testing/SKILL.md`
   - `.agents/skills/integration-testing/SKILL.md`
   - `.agents/skills/documentation/SKILL.md`
4. Keep generated or unrelated user changes intact. Do not revert files you did
   not intentionally modify.

## Project Facts

- Python: `>=3.12` from `pyproject.toml`.
- Source package: `src/simtools/`.
- Applications: `src/simtools/applications/`, installed as `simtools-*`
  commands through `[project.scripts]`.
- Unit tests: `tests/unit_tests/`, mirroring `src/simtools/`.
- Integration tests: `tests/integration_tests/config/*.yml`, executed through
  `tests/integration_tests/test_applications_from_config.py`.
- Test resources: `tests/resources/static` and `tests/resources/generated`.
- Documentation: Sphinx in `docs/source/`, built from MyST Markdown plus some
  RST autodoc pages.
- Changelog fragments: `docs/changes/<pr-number>.<type>.md`.

## Development Conventions

- Use `pathlib` for paths.
- Use double quotes for strings and docstrings.
- Use f-strings for formatting.
- Use logging for user/developer messages; do not use `print` in library code.
- Put useful exception text in the raised exception. When wrapping exceptions,
  use `raise ... from exc`.
- Avoid `logger.error` immediately before raising; it usually duplicates the
  failure.
- Use `astropy.units` for physical quantities.
- Validate CTAO names through existing helpers in `simtools.utils.names`.
- Use semantic model versions without a leading `v` in new configs.
- Do not add type hints to function signatures unless the surrounding module
  already deliberately uses them.
- Keep cognitive complexity below 15; extract private helpers before adding
  deeply nested logic.
- Keep code and docs ASCII-only unless an existing file clearly requires
  another character set.

## Testing

Default `pytest` runs unit tests only because `tool.pytest.ini_options.testpaths`
is `tests/unit_tests/`.

Use focused commands first:

```bash
pytest tests/unit_tests/path/to/test_module.py
pytest -vv tests/unit_tests/path/to/test_module.py::test_name
pytest --durations=10 tests/unit_tests/
```

Run broader checks when shared behavior changes:

```bash
pytest
pytest --cov=simtools --cov-report=term-missing
pre-commit run --all-files
```

Unit-test rules:

- Use plain pytest functions, not test classes.
- Cover changed success paths, error paths, and branches.
- Use local fixtures first; use `tests/unit_tests/conftest.py` for fixtures
  shared across unit-test modules.
- Shared repo fixtures such as `test_resources_path` and `simtools_root_path`
  live in `tests/conftest.py`.
- Use `tmp_test_directory` for file I/O. Do not introduce hardcoded `/tmp`,
  `tempfile`, or absolute temporary paths in tests.
- Mock databases, network calls, file I/O, CORSIKA, and sim_telarray in unit
  tests unless the test is explicitly marked for external resources.
- Use `pytest.approx()` for floats and
  `astropy.tests.helper.assert_quantity_allclose` for quantities.
- Warnings are treated as errors; fix deprecations instead of filtering them
  unless there is a clear project-wide reason.

## Integration Tests

Integration tests run real `simtools-*` applications from YAML configs. Use the
integration-testing skill for config work.

Important mechanics:

- Schema: `src/simtools/schemas/application_workflow.metaschema.yml`.
- Config shape starts with `applications:` and ends with
  `schema_name: application_workflow.metaschema` and `schema_version: 0.4.0`.
- `model_version_use_current: true` is lower-case and application-level.
- Generated paths such as `output_path`, `grid_output_path`, and
  `pack_for_grid_register` should be relative; the harness rewrites them into
  `tmp_test_directory`.
- Use `${static:path/to/file}` for maintained resources and
  `${generated:path/to/file}` for generated resources. Pytest resolves these
  against `--test_resources_path` / `--test-resources-path`, defaulting to
  `tests/resources`.
- Put `expected_sim_telarray_output` and `expected_sim_telarray_metadata`
  directly on the relevant `test_output_files` item.
- Use `test_simtel_cfg_files` for version-specific sim_telarray cfg
  comparisons.
- Add `docs.title` and `docs.summary` when the config should appear as a
  rendered documentation example.

Useful commands:

```bash
pytest --no-cov tests/integration_tests/test_applications_from_config.py
pytest -v -k "simtools-<app-name>" tests/integration_tests/test_applications_from_config.py
pytest -v -k "simtools-<app-name>_<test_name>" \
  tests/integration_tests/test_applications_from_config.py
pytest -v --model_version 6.0.2 -k "<test_name>" \
  tests/integration_tests/test_applications_from_config.py
pytest -v --test-resources-path /full/path/to/resources \
  tests/integration_tests/test_applications_from_config.py
```

Integration tests often require `.env` MongoDB settings and installed CORSIKA /
sim_telarray. Unit tests should not.

## Documentation

Use the documentation skill for docs, API reference, changelog, and docstring
work.

- Documentation pages are preferred in MyST Markdown.
- Application autodoc pages are small RST files in
  `docs/source/user-guide/applications/`.
- New application pages use one `.. automodule::` directive with `:members:`
  and usually `:exclude-members: main`.
- Add new application pages to `docs/source/user-guide/applications.md` in
  alphabetical order.
- Add new library modules to the relevant `docs/source/api-reference/*.md`
  file with `.. automodule:: <module.path>` and `:members:`.
- Public functions, classes, and methods need NumPy-style docstrings. Include
  only relevant sections such as `Parameters`, `Returns`, `Raises`, and
  `Examples`.
- Changelog fragment types: `feature`, `bugfix`, `api`, `doc`,
  `maintenance`, `model`. Use the PR number, not the issue number.

Docs validation:

```bash
cd docs
make clean
make html
make linkcheck
```

## Adding Code

New application checklist:

1. Add the application under `src/simtools/applications/`.
2. Register the command in `[project.scripts]` in `pyproject.toml`.
3. Add or update unit tests.
4. Add an integration config in `tests/integration_tests/config/`.
5. Add the RST application page and applications toctree entry.
6. Add a changelog fragment when working in a PR flow.

New library module checklist:

1. Add focused unit tests under the mirrored `tests/unit_tests/` path.
2. Add API reference documentation.
3. Add or update user documentation if behavior is user-facing.
4. Add a changelog fragment when working in a PR flow.

## Linting And Formatting

Pre-commit runs ruff, ruff-format, pylint, flake8 cognitive complexity,
docstring coverage, actionlint, pyproject-fmt, codespell, markdownlint,
yamllint, towncrier, and shellcheck.

Useful commands:

```bash
ruff check --fix
ruff format
pylint src/simtools/path/to/module.py
pre-commit run --all-files
```

Pylint excludes tests. Do not satisfy pylint by adding broad disables when a
small refactor or better naming fixes the issue.

## Domain Conventions

- Sites: `North`, `South`.
- Telescope names: examples include `LSTN-01`, `LSTS-01`, `MSTN-01`,
  `MSTS-05`; check `src/simtools/resources/array_elements.yml`.
- Array layout names and telescope names vary by model version; use
  `by_version` in integration configs where needed.
- Model-parameter schema changes can affect sim_telarray metadata. If an
  integration failure says a required metadata key is missing, inspect the
  relevant schema, DB/mock parameter data, and sim_telarray metadata registry
  before changing the test expectation.

## Recurring Failure Checks

These issues have appeared repeatedly in local Codex logs and CI snippets:

- Missing generated/static test resources: prefer `${static:...}` and
  `${generated:...}` over raw `tests/resources/...` paths in integration
  configs.
- Wrong output descriptor in integration checks: verify whether the file is
  under `output_path`, `grid_output_path`, or `pack_for_grid_register`.
- sim_telarray files not produced: inspect the application log path printed in
  stderr before changing expected filenames.
- `log_inspector` failures: look for real `error`, `exception`, `traceback`,
  `failed`, `runtime warning`, or `segmentation fault` text in stdout/stderr.
- Warnings-as-errors failures: update deprecated APIs, for example matplotlib
  colormap handling, rather than suppressing the warning locally.
- Pylint duplicate-code or complexity failures: extract a helper only when it
  improves readability and matches local patterns.
- `Undocumented module` CI failures: add the missing API reference entry.
