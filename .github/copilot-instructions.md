# Simtools - Copilot Instructions

This is a Python project for Monte Carlo simulations and high-energy gamma-ray astronomy, specifically for the Cherenkov Telescope Array Observatory (CTAO). The toolkit manages simulation model parameters, configures and runs simulation productions for arrays of imaging atmospheric Cherenkov telescopes.

## Project Overview

**simtools** is part of the CTAO SimPipe pipeline and provides:

- Simulation model libraries and management
- Database interfaces for simulation model parameters (MongoDB)
- Tools for preparing, configuring, running, and validating simulation productions
- Applications for deriving and validating model parameters
- Standardized interfaces for CORSIKA air shower simulations and sim_telarray telescope simulations
- I/O and reporting tools

**Python Version:** ≥ 3.12

## Project Structure

```
simtools/
├── src/simtools/           # Main package
│   ├── model/             # Telescope/camera/array models, calibration, database integration
│   ├── simtel/            # sim_telarray integration (config, events, ray tracing)
│   ├── corsika/           # CORSIKA air shower simulation integration
│   ├── ray_tracing/       # PSF analysis and mirror panel calculations
│   ├── camera/            # Camera-specific tools (efficiency, photon-electron spectrum)
│   ├── sim_events/        # Event data I/O (readers, writers, histograms)
│   ├── db/                # MongoDB database layer
│   ├── layout/            # Array geometry and telescope positions
│   ├── production_configuration/ # Simulation production grid generation
│   ├── visualization/     # Plotting tools (arrays, cameras, PSF, events)
│   ├── runners/           # Execution management (CORSIKA, sim_telarray, HTCondor)
│   ├── applications/      # 50+ CLI tools (installed as simtools-* commands)
│   ├── io/                # Data handling (table readers, ASCII handlers)
│   ├── data_model/        # Schema validation and metadata
│   ├── configuration/     # Argument parsing and configuration management
│   ├── job_execution/     # Job scheduling (HTCondor, process pools)
│   ├── reporting/         # Auto-report generation tools
│   ├── schemas/           # JSON/YAML schema files for validation
│   ├── testing/           # Test utilities and helpers
│   └── utils/             # Common utilities (geometry, value conversion, naming)
├── tests/
│   ├── unit_tests/        # Unit tests mirroring src/ structure
│   ├── integration_tests/ # Integration tests for applications
│   ├── conftest.py        # Shared pytest fixtures and configuration
│   └── resources/         # Test data and reference files
├── docs/                  # Sphinx documentation
│   ├── source/
│   │   ├── developer-guide/  # Development documentation
│   │   ├── user-guide/       # User documentation
│   │   ├── api-reference/    # Auto-generated API docs
│   │   ├── components/       # Component-level documentation
│   │   └── data-model/       # Data model documentation
│   └── Makefile           # Documentation build commands
├── database_scripts/      # Database management scripts
├── docker/                # Docker/Podman container definitions
├── pyproject.toml         # Project configuration, dependencies, tool settings
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── .env_template          # Template for environment variables
```

## Setup

```bash
# Option 1: pip (local development)
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,tests]'
pre-commit install

# Option 2: Containers (includes CORSIKA/sim_telarray)
podman run --rm -it -v "$(pwd)/external:/workdir/external" \
    ghcr.io/gammasim/simtools-dev:latest
```

**Environment variables** (`.env` from `.env_template`):
- `SIMTOOLS_DB_SERVER`: MongoDB server
- `SIMTOOLS_DB_API_USER`, `SIMTOOLS_DB_API_PW`: Credentials
- `SIMTOOLS_DB_SIMULATION_MODEL_VERSION`: e.g., v0.12.0
- `SIMTOOLS_CORSIKA_PATH`, `SIMTOOLS_SIM_TELARRAY_PATH`: Executable paths

## Testing

**MANDATORY:** Unit tests for all library code (target ≥90% coverage, aim for 100%).

```bash
pytest                                      # Run all unit tests (default testpath)
pytest -n 4                                 # Parallel
pytest --cov=simtools --cov-report=html    # With coverage
pytest --durations=10 tests/unit_tests/     # Find slow tests
pytest -vv tests/unit_tests/model/test_foo.py::test_bar  # Single test
pytest -s tests/unit_tests/                 # Show print output
pytest --no-cov tests/integration_tests/    # Integration tests
```

**Unit Test Guidelines:**
- Location: `tests/unit_tests/` mirror `src/simtools/` structure
- Use simple test functions (not test classes)
- Every function/method in library code MUST have test coverage
- Tests must be FAST (check with `--durations=10`)
- Fixtures: shared in `tests/conftest.py`, module-specific at top of file
- Use `tmp_test_directory` fixture for file I/O (NOT `tmp_path`)
- Mock external dependencies (DB, file I/O, network)
- Use `pytest.approx()` for float comparisons
- Use `astropy.tests.helper.assert_quantity_allclose` for units

**Integration Tests:** Full application workflows with real configs/data in `tests/integration_tests/config/`.

**Debug Tips:**
```bash
pytest --pdb tests/unit_tests/              # Drop into debugger on failure
pytest --random-order tests/unit_tests/     # Find test dependencies
pytest --lf tests/unit_tests/               # Re-run failed tests
```

## Code Standards

**Pre-commit (MANDATORY):**
```bash
pre-commit install               # Once after cloning
pre-commit run --all-files       # Before committing
```
Runs: ruff, pylint, docstring coverage (70%+), spell-check, markdown/yaml lint, etc.

**Key Style Rules:**
- Line length: 100 characters
- Quotes: double (`"""` for docstrings)
- Imports: sorted via ruff (isort rules, `I` ruleset)
- Linting: ruff (fast) + pylint (thorough)

```bash
ruff check --fix                 # Auto-fix issues
ruff format                      # Format code
pylint src/simtools/model/       # Check module
```

## Coding Conventions

**Python Style:**
- Use **pathlib** for file paths (NOT `os.path`)
- Use **f-strings** for formatting
- Use **logging** (NOT `print`)
- Use **astropy.units** for physical quantities
- Validate names with `simtools.utils.names` functions
- Use semantic versions without "v" prefix ("1.0.0", not "v1.0.0")
- Do not use **type hints** on function signatures

**Docstrings (MANDATORY):** NumPy style with Parameters, Returns, Raises, Examples sections. 70%+ coverage required for all public functions/methods. Private functions can have single-line docstring if self-explanatory.

```python
def example_function(parameter, optional_param):
    """Brief one-liner.

    Longer description if needed.

    Parameters
    ----------
    parameter : str
        Description.
    optional_param : int, optional
        Description.

    Returns
    -------
    dict
        Description.
    """
```

**Logging:** Use appropriate levels (all with f-strings):

```python
logger.info(f"Progress: {value}")        # General users
logger.warning(f"Issue: {value}")        # Users should know
logger.debug(f"Calculation: {value}")    # Developers only
logger.error(f"Failed: {value}")         # Exceptions/exit
```

**Naming Conventions:**

- Telescope names: `LSTN-01`, `MSTS-05` - see `src/simtools/resources/array_elements.yml`
- Site names: 'South', 'North'
- Constants: UPPER_CASE at module level

**Minimize redundant comments** — code should be self-explanatory.

## Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes, test frequently
pytest tests/unit_tests/model/

# 3. Lint and format
pre-commit run --all-files

# 4. Commit and push
git commit -m "Description"
git push origin feature/my-feature
```

## Adding a New Application

Every new application added to `src/simtools/applications/` requires:

1. **Register the CLI entry point** in `pyproject.toml` under `[project.scripts]`.
2. **Add a doc page** `docs/source/user-guide/applications/simtools-<app-name>.rst` following the pattern of existing files (one `.. automodule::` directive).
3. **Add a toctree entry** to `docs/source/user-guide/applications.md` in alphabetical order.
4. **Add an integration test config** in `tests/integration_tests/config/<app_name>_run.yml`.
5. **Add unit tests** in `tests/unit_tests/` mirroring the `src/` structure.

Every new **library module** added to `src/simtools/` (outside `applications/`) requires:

6. **Add an API reference entry** to the relevant `docs/source/api-reference/*.md` file using the `automodule` directive. The CI check `Undocumented module` will fail if this is missing.

## Opening a Pull Request

1. **Open a draft PR first** to get the PR number before adding the changelog fragment.
2. **Add the changelog fragment** using the PR number as the file name:
   `docs/changes/<pr-number>.<type>.md` (one line, concise description).
3. **Commit and push** the changelog fragment to the same branch.
4. **PR title and body** should be short and clear:
   - Title: imperative mood, under 72 characters (e.g. `Fix mirror PSF calculation`).
   - Body: bullet list of what changed and why; no filler text.
5. **Mark ready for review** only when tests pass and pre-commit is clean.

## Troubleshooting

**CORSIKA/sim_telarray not found:**
- Use container environment (recommended)
- Or set `SIMTOOLS_CORSIKA_PATH` and `SIMTOOLS_SIM_TELARRAY_PATH` in `.env`
- Unit tests mock these automatically

**Database connection failures:**
- Check `.env` credentials
- Verify MongoDB running locally
- CI uses local MongoDB service

**Pre-commit failures:**
- Run `pre-commit run --all-files` to see all issues
- Use `ruff check --fix` to auto-fix

**Import errors:**
- Use `pip install -e .` (editable mode)
- Activate correct environment
- Check Python ≥ 3.12

**Test failures:**
- Integration tests need `tests/resources/`
- Some tests marked `uses_model_database` need DB access
- Unit tests should mock external dependencies

## Documentation

```bash
cd docs
make html                   # Build HTML docs
make linkcheck              # Check links
```

**Version:** Managed by `setuptools-scm` (git-based). DO NOT edit `src/simtools/_version.py`.

**Changelog:**
- Add fragments to `docs/changes/<pr-number>.<type>.md` (types: feature, bugfix, api, doc, maintenance, model).
- Changelogs should not exceed 1 line.
- Use pull request IDs (not issue IDs) as the fragment number.

**API Reference:**
- Every new module must be added to `docs/source/api-reference/` — either as a new `.md` file (listed in `index.md`) or as a new `## section` in the relevant existing `.md` file, using `.. automodule::` with `:members:`.



## Key Dependencies

**Core:** numpy, scipy, astropy | **Data:** pymongo, jsonschema, pyyaml | **Viz:** matplotlib, adjusttext | **Dev:** pytest, ruff, pylint, pre-commit, sphinx

## AI Agent Roles & Responsibilities

### 🔬 Astrophysics Expert Agent

**Focus:** Scientific correctness, physical models, numerical stability.

**Key Rules:**
- Validate all physical quantities use correct units (astropy.units)
- Check numerical accuracy for floating-point comparisons (use `pytest.approx()`)
- Verify astronomical conventions (coordinate systems, telescope naming)
- Validate CORSIKA/sim_telarray integration matches physics models
- Ensure Monte Carlo statistical methods are correct
- Document assumptions in code and docstrings

**When implementing:** Study `src/simtools/model/` patterns first.

---

### 💻 Developer Agent

**Focus:** Code quality, architecture, testing, maintainability.

**Key Rules:**
1. **Always test:** `pytest tests/unit_tests/` after changes (≥90% coverage)
2. **Always lint:** `pre-commit run --all-files` before commits
3. **Follow conventions:** pathlib, logging, f-strings, NumPy docstrings
4. **Error messages:** no `logger.error`, put the error message into the error (e.g. `ValueError("Invalid type")`) and always do `from exc`
5. **Mock external deps:** DB, file I/O, network must be mocked in unit tests
6. **Use tmp_test_directory** for file I/O (NOT `tmp_path` or `/tmp/`)
7. **Do not hardcode `/tmp/` paths in tests** (including CLI argument values); always derive test paths from `tmp_test_directory`
8. **Study patterns:** Check existing code before implementing
9. **Document:** NumPy-style docstrings, 70%+ coverage required
10. **Make minimal changes:** Understand codebase first
11. **No premature optimization:** Clarity > speed
12. **Golden Rule:** If code is hard to understand, refactor it
13. **Cognitive Complexity:** Keep it below 15

**Validation:** 100% statement coverage for library code.

---

### 📚 Documentation Manager Agent

**Focus:** Clarity, completeness, consistency, user experience.

**Key Rules:**
- NumPy docstrings are **MANDATORY**, with the exception of private functions (where a single-line docstring is acceptable if self-explanatory).
- Every function/class/method MUST have a docstring. Private functions can have single-line docstring if self-explanatory.
- Include Parameters, Returns, Raises, Examples sections
- Use clear, concise language (avoid jargon)
- Update docs when changing APIs
- Add changelog fragments to `docs/changes/<issue>.<type>.md`
- Validate all links: `cd docs && make linkcheck`
- Keep examples runnable and tested
- Maintain consistency with existing documentation style

**When writing:** Make examples reflect real use cases from `tests/integration_tests/`.

---

## Core Principles (All Agents)

✅ **DO:**
- Write tests for everything (unit tests are mandatory)
- Run `pre-commit run --all-files` before every commit
- Use simple, readable code over clever optimizations
- Study existing patterns before implementing
- Document assumptions and non-obvious decisions
- import statements should be at the top of the file (even for unit test files)

❌ **DON'T:**
- Do not add inconsistent import statements or random package imports.
- Do not use non-ascii characters in code or docstrings (e.g., fancy quotes, degree symbols)
- Skip tests or claim "obvious code doesn't need tests"
- Hardcode file paths or credentials
- Add trivial comments ("# add five" before `x + 5`)
- Mix concerns in single functions
- Commit without passing linters and tests

**Quality Standard:** Correct (well-tested) → Readable (clear intent) → Documented (thorough) → Maintainable (follows conventions)
