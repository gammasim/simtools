# simtools Agent Guide

This file gives repo-wide instructions for AI agents working on simtools.

simtools is a Python toolkit for CTAO Monte Carlo production support: model
parameter handling, MongoDB access, CORSIKA and sim_telarray configuration,
application workflows, validation, reporting, and plotting.

In general, do not pretend you are a human developer. You are a tool, you don't think.
If given a task, follow the instructions and do not make assumptions. Just do what you are told.
If you are unsure, ask for clarification. Try to shut up.

## First Steps

1. Inspect the existing code and tests before changing behavior.
2. Prefer the narrowest change that fixes the requested issue.
3. Make architectural changes for the long term. Avoid short-term hacks.
4. Check whether a local skill applies:
   - `.agents/skills/unit-testing/SKILL.md`
   - `.agents/skills/integration-testing/SKILL.md`
   - `.agents/skills/documentation/SKILL.md`
5. Keep generated or unrelated user changes intact. Do not revert files you did
   not intentionally modify.

## Project Facts

- Python: `>=3.14` from `pyproject.toml`.
- Python 3.14 allows multiple exception types without parentheses (PEP 758) when not using `as`, e.g. `except AttributeError, KeyError, TypeError:`; use parentheses for `except (AttributeError, KeyError, TypeError) as exc:`.
- Source package: `src/simtools/`.
- Applications: `src/simtools/applications/`, installed as `simtools-*`
  commands through `[project.scripts]`.
- Unit tests: `tests/unit_tests/`, mirroring `src/simtools/`.
- Integration tests: `tests/integration_tests/config/*.yml`, executed through
  `tests/integration_tests/test_applications_from_config.py`.
- Test resources: versioned integration resources in `simtools-tests`; ordinary
  unit tests must not depend on repository-local or external resource files.
- Documentation: Sphinx in `docs/source/`, built from MyST Markdown plus some
  RST autodoc pages.
- Changelog fragments: `docs/changes/<pr-number>.<type>.md`.

## Development Conventions

- Use `pathlib` for paths.
- Do not add `__init__.py` files; this project uses implicit namespace packages.
- Use double quotes for strings and docstrings.
- Use f-strings for formatting.
- Use logging for user/developer messages; do not use `print` in library code.
- Documentation and comments should describe current behavior, not historical
  production details or changes from earlier behavior.
- Put useful exception text in the raised exception. When wrapping exceptions,
  use `raise ... from exc`.
- Avoid `logger.error` immediately before raising; it usually duplicates the
  failure.
- Use `astropy.units` for physical quantities.
- Validate CTAO names through existing helpers in `simtools.utils.names`.
- When introducing a new schema version in `src/simtools/schemas`, add a new
  YAML document and preserve the existing version; do not replace it.
- Use semantic model versions without a leading `v` in new configs.
- Do not add type hints to function signatures unless the surrounding module
  already deliberately uses them.
- Keep cognitive complexity below both configured limits: flake8-cognitive-
  complexity is capped at 15 and Ruff's mccabe check is capped at 12. Extract
  small private helpers before adding branches to an already complex function.
- Do not add nested conditional expressions, implicit adjacent string literals
  inside collections, redundant subclass/base exception entries, unused
  function parameters, or duplicated long string literals. These patterns are
  repeatedly rejected by Ruff, pylint, or SonarQube. Use an explicit branch,
  explicit string construction, a single appropriate exception type, a
  meaningful parameter (or the repository's accepted ignored-parameter form),
  and a named constant respectively.
- Keep code and docs ASCII-only unless an existing file clearly requires
  another character set.

## Container Workflows

- If a task requires building or running containers, first check whether Docker,
  Podman, or Apptainer is available and use an available runtime where
  practical.
- Do not install or require a container runtime without explicit user approval.

## Python Environment

- If `python`, `pytest`, or `pylint` is not available in the current
  environment, check for a Conda or Mamba environment named `simtools-dev` and
  run the command there when available.

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
- Every Python module under `src/simtools/` must have a matching unit-test
  file under `tests/unit_tests/`, preserving its package directory and using
  the `test_<module>.py` name, except for the modules excluded by CI:
  `__init__.py`, `_version.py`, and anything under `applications/`. This
  includes small helper modules such as `simtel/pulse_shapes.py` and
  `utils/value_conversion.py`; do not leave them uncovered or rely on tests of
  a neighboring module.
- Use local fixtures first; use `tests/unit_tests/conftest.py` for fixtures
  shared across unit-test modules.
- Shared repo fixtures such as `test_resources_path` and `simtools_root_path`
  live in `tests/conftest.py`.
- Do not make ordinary unit tests depend on checked-in, downloaded, or external
  files. If file parsing or writing is the behavior under test, generate the
  smallest valid input in `tmp_test_directory` within the test.
- Keep file-format compatibility and resource-heavy checks in integration tests.
- Use `tmp_test_directory` for file I/O. Do not introduce hardcoded `/tmp`,
  `tempfile`, or absolute temporary paths in tests.
- Mock databases, network calls, file I/O, CORSIKA, and sim_telarray in unit
  tests unless the test is explicitly marked for external resources.
- Use `pytest.approx()` for floats and
  `astropy.tests.helper.assert_quantity_allclose` for quantities.
- Aim for 90--95% line and branch coverage for changed non-application code;
  SonarQube checks coverage on new code. Add tests for meaningful success,
  failure, and boundary branches instead of padding coverage with tests that
  have no independent oracle.
- Warnings are treated as errors; fix deprecations instead of filtering them
  unless there is a clear project-wide reason.
- Run tests against this checkout, not an installed or editable copy from a
  different worktree. Before diagnosing an import-dependent failure, verify
  `python -c "import simtools; print(simtools.__file__)"`; when using Conda,
  prepend this checkout's `src` with `env PYTHONPATH="$PWD/src"`.

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
- Set `SIMTOOLS_TESTS_PATH` and `SIMTOOLS_TESTS_TAG`, or use
  `--simtools_tests_tag`, to select a tagged `simtools-tests` resource
  bundle when `SIMTOOLS_TEST_RESOURCES` or `--test_resources_path` is not
  provided.
- Use `${static:path/to/file}` for maintained resources and
  `${generated:path/to/file}` for generated resources. Pytest resolves these
  against `--test_resources_path`, defaulting to `SIMTOOLS_TEST_RESOURCES` or
  `--test_resources_path`.
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
pytest -v --test_resources_path /full/path/to/resources \
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
- Do not add development plans to the documentation.

Docs validation:

```bash
conda run -n simtools-dev env PYTHONPATH="$PWD/src" make -C docs clean html linkcheck
```

The `docs/Makefile` enables `-W -n --keep-going`, so treat every Sphinx
warning as a failure. The build imports the checkout's applications for CLI
help and autodoc; the explicit `PYTHONPATH` is required when an installed
simtools package may otherwise win. Check the import location before a build
if the traceback mentions missing attributes, stale code, or a syntax error in
`site-packages`.

When adding or changing documentation:

- Keep application modules to a one-line synopsis and put operational detail,
  examples, inputs, outputs, and CLI explanation in the application's MyST
  page. Do not duplicate the same description in the module docstring and the
  page.
- Add every new application page to the applications toctree and every new
  library module to the relevant API-reference page. A file existing under
  `docs/source/` is not enough; `toc.not_included` is a documentation failure.
- Use the exact module path expected by neighboring `automodule` entries (for
  example, `model_repository.reader`), not only a basename. Ensure the path is
  importable from the checkout and that public docstring parameters, returns,
  and types match the actual signature.
- Resolve new Sphinx cross-reference warnings at their source. Do not add a
  broad `nitpick_ignore` or `suppress_warnings` entry to hide an unresolved
  reference.

Before handing off a change that adds, removes, or moves a library module, run
the API coverage check as well. Every reported module must be added to the
appropriate API reference page with an `automodule` entry for its complete
module path, matching the import style already used by that API page:

```bash
python - <<'PY'
from pathlib import Path

src_root = Path("src/simtools")
api_text = "\n".join(
    path.read_text(encoding="utf-8")
    for path in Path("docs/source/api-reference").glob("*.md")
)
missing = []
for path in src_root.rglob("*.py"):
    relative = path.relative_to(src_root)
    if (
        path.name in {"__init__.py", "_version.py"}
        or relative.parts[0] == "applications"
        or relative.parts[0].startswith("_")
    ):
        continue
    module = ".".join(relative.with_suffix("").parts)
    candidates = (module, f"simtools.{module}")
    if not any(f".. automodule:: {candidate}" in api_text for candidate in candidates):
        missing.append(module)
if missing:
    raise SystemExit("Undocumented modules:\n" + "\n".join(sorted(missing)))
PY
```

## Adding Code

New application checklist:

1. Add the application under `src/simtools/applications/`.
2. Register the command in `[project.scripts]` in `pyproject.toml`.
3. Add or update unit tests.
4. Add an integration config in `tests/integration_tests/config/`.
5. Add the RST application page and applications toctree entry.
6. Add a changelog fragment when working in a PR flow.
7. Applications in `src/simtools/applications/` are entry scripts. If possible, avoid
   adding code here and instead put reusable code in a library module.

New library module checklist:

1. Add focused unit tests under the mirrored `tests/unit_tests/` path.
2. Add API reference documentation.
3. Add or update user documentation if behavior is user-facing.
4. Add a changelog fragment when working in a PR flow.

Before handing off a change that adds, removes, or moves a library module, run
the same source-to-test check used by CI and resolve every reported path:

```bash
python - <<'PY'
from pathlib import Path

src_root = Path("src/simtools")
test_root = Path("tests/unit_tests")
missing = []
for path in src_root.rglob("*.py"):
    relative = path.relative_to(src_root)
    if path.name in {"__init__.py", "_version.py"} or relative.parts[0] == "applications":
        continue
    if not (test_root / relative.parent / f"test_{path.stem}.py").exists():
        missing.append(str(relative))
if missing:
    raise SystemExit("Modules without unit tests:\n" + "\n".join(sorted(missing)))
PY
```

## Linting And Formatting

Pre-commit runs ruff, ruff-format, pylint, flake8 cognitive complexity,
docstring coverage, actionlint, pyproject-fmt, codespell, markdownlint,
yamllint, towncrier, and shellcheck.

Useful commands:

```bash
pre-commit run --all-files
pre-commit run --files path/to/changed.py
ruff check path/to/changed.py
ruff format --check path/to/changed.py
flake8 --select=CCR001 --max-cognitive-complexity=15 path/to/changed.py
pylint -rn -sn --init-hook="import sys; sys.path.insert(0, 'src')" \
  src/simtools/path/to/module.py
git diff --check
```

Use the same Python environment and checkout for all commands. If a tool is
missing, use the `simtools-dev` environment and set `PYTHONPATH` to this
checkout's `src`; do not silently lint an installed package. `ruff check --fix`
and pre-commit may modify files: inspect the diff and rerun the complete
pre-commit suite after automatic fixes. A focused check is useful during
iteration, but the final check must match CI (`pre-commit run --all-files`).

Pylint excludes tests. Do not satisfy pylint, Ruff, flake8, or SonarQube with
broad disables or generated boilerplate when a small refactor, a correct
import path, or a better name fixes the problem. In particular, keep helper
functions simple enough for both the Ruff and flake8 complexity limits, and
keep line length at 100 characters.

## SonarQube And Workflow Quality

Treat SonarQube findings as defects to prevent during implementation, not as
post-merge clean-up. Before changing a workflow, run actionlint, yamllint, and
shellcheck through pre-commit and inspect the complete workflow diff. Apply
these rules:

- Pin every external GitHub Action to a full commit SHA; a tag, branch, or
  floating reference triggers the security-hotspot check.
- Never expand `${{ secrets.* }}` directly in a `run:` block. Pass secrets in
  the step's `env` and reference the environment variable in the shell. Keep
  shell expansions quoted, use `set -euo pipefail` where appropriate, and pass
  scanner or command arguments as an array when possible.
- Keep default workflow permissions restrictive (`permissions: {}` when
  possible) and declare required permissions at the job that uses them. Do not
  put a read permission only at workflow level when SonarQube expects a
  job-level declaration.
- Remove unused callback or fixture parameters. If an API requires a
  parameter, use the repository's accepted ignored-parameter naming and verify
  both pylint and SonarQube before handoff.
- Do not use nested ternaries; spell out the branch. Use `\d` instead of
  `[0-9]` in regular expressions where equivalent, define constants for
  repeated literals, and do not catch a subclass together with its base class.
- Use SHA-256 for file or configuration digests unless a non-security use of a
  different algorithm is explicitly justified in code. A digest used for
  integrity or trust must not rely on SHA-1.
- Keep new-code coverage near 90--95% and inspect the coverage report for
  uncovered conditions, not just the total percentage. SonarQube's new-code
  metric can fail even when the repository-wide percentage looks healthy.

Do not assume Ruff and SonarQube are interchangeable. When their suggestions
appear to conflict, preserve clear behavior, make the transformation explicit,
and run both tools plus the relevant tests before choosing a suppression.

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
- Pylint unused-argument/import/no-member failures: first confirm that pylint
  is analyzing this checkout with the `src` init hook; then fix the import or
  remove/rename the unused argument. Do not add a broad pylint disable.
- Ruff and flake8 complexity failures: split the function at a meaningful
  responsibility boundary and rerun both complexity checks; do not merely
  move branches or add a suppression.
- SonarQube workflow findings: use full action SHAs, step-level secret `env`,
  and job-level permissions as described above. Review every workflow touched
  by a shared CI change.
- SonarQube maintainability findings: avoid nested conditional expressions,
  duplicate literals, unused parameters, redundant exception classes, and
  unclear comprehension rewrites.
- `Undocumented module` CI failures: add the missing API reference entry.
- The API documentation check scans every non-application, non-private
  `src/simtools/**/*.py` module by basename. When adding or moving a module,
  ensure that its basename is present in the appropriate
  `docs/source/api-reference/*.md` page, with an `automodule` directive for
  the complete module path (not merely a coincidental basename match). Resolve
  every `Undocumented module: <name>` result; do not silence the check or rely
  on a partial documentation build.
