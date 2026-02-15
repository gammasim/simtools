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
│   │   └── api-reference/    # Auto-generated API docs
│   └── Makefile           # Documentation build commands
├── database_scripts/      # Database management scripts
├── docker/                # Docker/Podman container definitions
├── pyproject.toml         # Project configuration, dependencies, tool settings
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── .env_template          # Template for environment variables
└── environment.yml        # Conda/mamba environment definition
```

## Setup

```bash
# Option 1: pip (local development)
pip install -e '.[dev,tests]'
pre-commit install

# Option 2: Conda/Mamba
mamba env create -f environment.yml
mamba activate simtools-dev
pip install -e .

# Option 3: Containers (includes CORSIKA/sim_telarray)
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
pytest tests/unit_tests/                    # Run all
pytest -n 4 tests/unit_tests/               # Parallel
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
- Imports: sorted with `isort`
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
- Do not use  **type hints** on function signatures

**Docstrings (MANDATORY):** NumPy style with Parameters, Returns, Raises, Examples sections. 70%+ coverage required for all public functions/methods. Private function can have single-line docstring if self-explanatory.

```python
def example_function(parameter: str, optional_param: int | None = None) -> dict:
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

**Logging:** Use appropriate levels:
logger.info(f"Progress: {value}")        # General users
logger.warning(f"Issue: {value}")        # Users should know
logger.debug(f"Calculation: {value}")    # Developers
logger.error(f"Failed: {value}")         # Exceptions/exit

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

**Changelog:** Add fragments to `docs/changes/<issue>.<type>.md` (types: feature, bugfix, api, doc, maintenance, model).



## Key Dependencies

**Core:** numpy, scipy, astropy | **Data:** pymongo, jsonschema, pyyaml | **Viz:** matplotlib, adjusttext | **Dev:** pytest, ruff, pylint, pre-commit, sphinx

## For AI Coding Agents

**Principles:** Prioritize **simplicity** and **maintainability** over cleverness.

1. **Always test:** `pytest tests/unit_tests/` after changes (≥90% coverage)
2. **Always lint:** `pre-commit run --all-files` before commits
3. **Follow conventions:** pathlib, logging, f-strings, type hints, NumPy docstrings
4. **Use mocking:** External dependencies (DB, file I/O, network) must be mocked in unit tests
5. **Validate names:** Use `simtools.utils.names` functions
6. **Study patterns:** Check existing code for conventions before implementing
7. **Document thoroughly:** NumPy-style docstrings, 70%+ coverage required
8. **Make minimal changes:** Understand the codebase structure first
9. **Write readable code:** Prefer clarity over optimization; avoid premature optimization
10. **Use containers:** When working with CORSIKA/sim_telarray

**Golden Rule:** If your change makes the code harder to understand, refactor it.

This is a scientific computing project with high quality standards. Code **must be**: correct (well-tested), readable (clear intent), documented (thorough), and maintainable (follows conventions).
