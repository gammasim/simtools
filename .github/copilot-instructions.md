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

**Python Version:** ≥ 3.12 (supports 3.12, 3.13, 3.14)

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

## Installation & Setup

### Quick Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/gammasim/simtools.git
cd simtools

# Install with development dependencies
pip install -e '.[dev,tests]'

# Install pre-commit hooks (IMPORTANT for development)
pre-commit install
```

### Using Conda/Mamba

```bash
# Create environment from file
mamba env create -f environment.yml
mamba activate simtools-dev
pip install -e .
```

### Using Containers (Strongly Recommended for Developers)

Containers include CORSIKA, sim_telarray, and all dependencies:

```bash
podman run --rm -it -v "$(pwd)/external:/workdir/external" \
    ghcr.io/gammasim/simtools-dev:latest \
    bash -c "source /workdir/env/bin/activate && cd /workdir/external/simtools && pip install -e . && bash"
```

### Environment Variables

Copy `.env_template` to `.env` and configure:

```bash
cp .env_template .env
# Edit .env with your database credentials and paths
```

Key environment variables:

- `SIMTOOLS_DB_SERVER`: MongoDB server address
- `SIMTOOLS_DB_API_USER`, `SIMTOOLS_DB_API_PW`: Database credentials
- `SIMTOOLS_DB_SIMULATION_MODEL`: Model name (default: CTAO-Simulation-Model)
- `SIMTOOLS_DB_SIMULATION_MODEL_VERSION`: Model version (e.g., v0.12.0)
- `SIMTOOLS_CORSIKA_PATH`: Path to CORSIKA installation
- `SIMTOOLS_SIM_TELARRAY_PATH`: Path to sim_telarray installation

## Testing

**IMPORTANT:** Unit tests are MANDATORY for all library code (target: >90% coverage, aim for ~100%).

### Running Tests

```bash
# Run all unit tests (with coverage)
pytest tests/unit_tests/

# Run tests in parallel (faster)
pytest -n 4 tests/unit_tests/

# Run specific test file
pytest tests/unit_tests/model/test_telescope_model.py

# Run with coverage report
pytest --cov=simtools --cov-report=html tests/unit_tests/

# Check test durations to find slow tests
pytest --durations=10 tests/unit_tests/

# Run integration tests (requires CORSIKA/sim_telarray)
pytest --no-cov tests/integration_tests/

# Test specific application
pytest -v -k "simtools-plot-array-layout" tests/integration_tests/test_applications_from_config.py
```

### Test Guidelines

**Unit Tests:**

- Location: `tests/unit_tests/` with structure mirroring `src/simtools/`
- Use simple test functions, NOT test classes
- Every function/method in library code MUST have tests
- Sort tests in same order as functions in the module
- Tests must be FAST (check with `--durations=10`)
- Use mocking for external dependencies (DB, file I/O, network)
- Use `tmp_test_directory` fixture for file I/O (NOT `tmp_path` or `tempfile`)
- Use `pytest.approx()` for floating point comparisons
- Use `astropy.tests.helper.assert_quantity_allclose` for astropy quantities

**Fixtures:**

- Shared fixtures in `tests/conftest.py` (must have docstrings)
- Module-specific fixtures at top of test file
- List all fixtures: `pytest --fixtures`

**Integration Tests:**

- Location: `tests/integration_tests/`
- Test full applications with typical use cases
- Configuration files in `tests/integration_tests/config/`
- Can validate output files, compare with references, check patterns

### Test Markers

```python
@pytest.mark.uses_model_database  # Test uses model parameter database
```

## Linting & Code Quality

### Pre-commit Hooks (MANDATORY)

```bash
# Install hooks (do this once after cloning)
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Skip pre-commit (rarely needed)
git commit --no-verify
```

Pre-commit runs:

- **ruff** (formatting and linting)
- **pylint** (code analysis)
- **docstr-coverage** (70%+ docstring coverage required)
- **codespell** (spell checking)
- **markdownlint** (Markdown linting)
- **yamllint** (YAML linting)
- **actionlint** (GitHub Actions linting)
- **shellcheck** (shell script linting)
- Standard checks (trailing whitespace, end-of-file-fixer, etc.)

### Linting Commands

```bash
# Run ruff (fast linter and formatter)
ruff check                    # Check for issues
ruff check --fix              # Auto-fix issues
ruff format                   # Format code

# Run pylint
pylint $(git ls-files 'src/simtools/*.py')

# Check specific file
ruff check src/simtools/model/telescope_model.py
pylint src/simtools/model/telescope_model.py

# Validate pyproject.toml
pip install validate-pyproject
validate-pyproject pyproject.toml
```

### Code Style Configuration (pyproject.toml)

**Ruff settings:**

- Line length: 100
- Indent: 4 spaces
- Quote style: double quotes
- Docstring style: NumPy
- Enabled rules: F (pyflakes), I (isort), N (pep8-naming), D (pydocstyle), PTH (use pathlib), NPY (numpy), UP (pyupgrade), RET (return statements), and more
- Ignored: D (documentation) for test files

**Pylint settings:**

- Line length: 100
- Max module lines: 1500
- Accepted short variable names: `i, j, k, x, y, n, f, r, ex, db, im, sh, ax, ce, xx, yy, zz, C1-C4, N1-N4, lst, mst, sst, sct, hess, magic, veritas`
- Disabled checks: `missing-module-docstring`, `logging-fstring-interpolation`, `fixme`

## Coding Conventions

### General Python Guidelines

**DO:**

- Use **pathlib** for file paths, NOT `os.path`
- Use **logging** for output, NOT `print`
- Use **f-strings** for string formatting
- Use semantic versions without "v" prefix (e.g., "1.0.0", not "v1.0.0")
- Use **astropy.units** for physical quantities
- Validate names using functions in `simtools.utils.names`
- Minimize comments (code should be self-explanatory)

**DON'T:**

- Hardcode file paths
- Add trivial comments (e.g., "# Adding five" before `add_five()`)
- Use `os.path` when `pathlib` is available
- Use `print()` for logging (use `logger.info()`, etc.)
- Start docstrings or replies with "This is the final outcome"

### Docstrings (MANDATORY)

Use **NumPy style** for all public functions, classes, and methods:

```python
def example_function(parameter, optional_param=None):
    """
    Brief description of what the function does.

    Longer description if needed, explaining behavior, algorithms, or important details.

    Parameters
    ----------
    parameter : type
        Description of parameter.
    optional_param : type, optional
        Description of optional parameter.

    Returns
    -------
    type
        Description of return value.

    Raises
    ------
    ExceptionType
        Description of when this exception is raised.

    Examples
    --------
    >>> example_function(42)
    42
    """
    ...
```

**Reference:** [NumPy Docstring Format](https://numpydoc.readthedocs.io/en/latest/format.html)

### Logging

Use appropriate log levels:

- **INFO**: Progress, results, input/output for general users
- **WARNING**: Issues users should know but can't change
- **DEBUG**: Information for developers/debugging
- **ERROR**: Issues leading to exceptions or program exit

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Processing telescope %s", telescope_name)
logger.debug("Intermediate calculation result: %s", result)
logger.warning("Using default value for missing parameter: %s", param)
logger.error("Failed to load configuration file: %s", filename)
```

### Naming Conventions

**Telescope Names:** Follow pattern "Site-Class-Type"

- Site: "North" or "South" (also accepts "paranal", "lapalma", "south", "north")
- Class: "LST", "MST", "SCT", or "SST"
- Type: Single number for real telescopes (e.g., "1"), or string with "D" for designs (e.g., "D234", "FlashCam-D")

Examples: `North-LST-1`, `North-LST-D234`, `North-MST-FlashCam-D`, `South-MST-NectarCam-D`

**Validation:** Use `simtools.utils.names.validate_array_element_name()` to validate names.

### Camera Conventions

- Camera pixel positions are typically in **centimeters (cm)**, not meters
- Use `camera.get_edge_pixels()` to get edge pixel indices
- Camera `rotate_angle` field stores original file value in radians; +90° is applied during pixel rotation but not stored back

### Test Documentation

- Test files should include a module-level docstring describing what tests are covered

### Default Values

- Use named constants at module level for default values in functions, especially plotting functions

## Build & Documentation

### Building Documentation

```bash
cd docs
make html           # Build HTML documentation
make linkcheck      # Check all links
```

Documentation is built with:

- **Sphinx** with **pydata-sphinx-theme**
- **myst-parser** for Markdown support
- **numpydoc** for NumPy-style docstrings
- **sphinx-design** for enhanced layouts

### Version Management

- Version is managed by **setuptools-scm** (git-based versioning)
- Version written to `src/simtools/_version.py` (DO NOT edit manually)

### Changelog

Uses **towncrier** for changelog management:

- Add changelog fragments in `docs/changes/`
- Fragment format: `<issue_number>.<type>.md`
- Types: `feature`, `bugfix`, `api`, `doc`, `maintenance`, `model`

```bash
# Generate changelog
towncrier build --version X.Y.Z
```

## CI/CD Workflows

Located in `.github/workflows/`:

- **CI-unittests.yml**: Unit tests on Python 3.12 & 3.13, with/without conda, code coverage, SonarQube analysis
- **CI-integrationtests.yml**: Integration tests with CORSIKA/sim_telarray
- **CI-linter.yml**: Pre-commit hooks, validation of pyproject.toml, CITATION.cff, non-ASCII check
- **CI-docs.yml**: Documentation build
- **CI-schema-validation.yml**: Validate JSON schemas
- **build-*.yml**: Docker/container image builds
- **changelog.yml**: Changelog validation
- **pypi.yml**: PyPI release automation

### Running CI Locally

```bash
# Linting (same as CI)
pre-commit run --all-files

# Unit tests (similar to CI, but CI uses MongoDB service)
pytest -n 4 --cov=simtools --cov-report=xml tests/unit_tests/

# Integration tests (requires CORSIKA/sim_telarray)
pytest --no-cov tests/integration_tests/
```

## Common Commands & Workflows

### Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes, run tests frequently
pytest tests/unit_tests/model/  # Test specific module

# 3. Run linters before commit
ruff check --fix
ruff format
pre-commit run --all-files

# 4. Commit (pre-commit hooks run automatically)
git add .
git commit -m "Description of changes"

# 5. Push and create PR
git push origin feature/my-feature
```

### Running Applications

All applications are installed as `simtools-*` commands:

```bash
# List all available commands
ls $(dirname $(which python))/../bin/simtools-*

# Example: Plot array layout
simtools-plot-array-layout --help
simtools-plot-array-layout --site North --array_layout alpha

# Example: Validate camera FOV
simtools-validate-camera-fov --telescope LSTN-01 --site North

# Example: Database operations
simtools-db-get-parameter-from-db --db_name CTA-Simulation-Model --parameter_name mirror_diameter
```

### Profiling Tests

```bash
# Identify slow tests
pytest --durations=10

# Profile specific test
pytest --no-cov --profile tests/unit_tests/utils/test_general.py

# Generate flame graphs (requires graphviz)
pytest --no-cov --profile-svg tests/unit_tests/utils/test_general.py
```

## Database Integration

### MongoDB Connection

The project uses MongoDB for simulation model storage. Connection configured via environment variables (see Environment Variables section).

### Database CLI Tools

```bash
# Upload model repository
simtools-db-upload-model-repository --db_simulation_model <name> --db_simulation_model_version <version>

# Get parameter from DB
simtools-db-get-parameter-from-db --parameter_name <name>

# Add file to DB
simtools-db-add-file-to-db --file <path>

# Inspect databases
simtools-db-inspect-databases
```

## Known Issues & Workarounds

### Common Issues

1. **CORSIKA/sim_telarray not found:**
   - Use container environment (recommended)
   - Or set `SIMTOOLS_CORSIKA_PATH` and `SIMTOOLS_SIM_TELARRAY_PATH` in `.env`
   - Unit tests mock these paths automatically (see `conftest.py`)

2. **Database connection failures:**
   - Check `.env` file has correct credentials
   - Verify MongoDB service is running (for local testing)
   - CI uses local MongoDB service (see `CI-unittests.yml`)

3. **Pre-commit hook failures:**
   - Run `pre-commit run --all-files` to see all issues
   - Use `--fix` with ruff to auto-fix many issues
   - Some hooks may need manual fixes (e.g., docstring coverage)

4. **Import errors after installation:**
   - Ensure you're using `pip install -e .` (editable mode)
   - Activate correct environment (`simtools-dev`)
   - Check Python version ≥ 3.12

5. **Test failures due to missing resources:**
   - Integration tests require test resources in `tests/resources/`
   - Some tests are marked with `uses_model_database` and need DB access
   - Unit tests should not require external resources (use mocking)

### Debugging Tips

```bash
# Run single test with verbose output
pytest -vv tests/unit_tests/model/test_telescope_model.py::test_specific_function

# Run with print statements visible
pytest -s tests/unit_tests/model/test_telescope_model.py

# Drop into debugger on failure
pytest --pdb tests/unit_tests/model/test_telescope_model.py

# Run tests in random order (find test dependencies)
pytest --random-order tests/unit_tests/

# Re-run failed tests
pytest --lf tests/unit_tests/
```

## Key Dependencies

**Scientific Computing:**

- numpy, scipy, astropy (core scientific libraries)
- h5py (HDF5 file format)
- boost-histogram (fast histograms)
- particle (particle properties)
- eventio (Cherenkov telescope data format)
- ctao-dpps-cosmic-ray-spectra (cosmic ray spectra)

**Data & Configuration:**

- pymongo (MongoDB client)
- jsonschema (JSON validation)
- pyyaml, toml (configuration formats)
- pyproj (coordinate transformations)
- python-dotenv (environment variables)

**Visualization:**

- matplotlib (plotting)
- adjusttext (label positioning)

**Development:**

- pytest (+ plugins: cov, xdist, mock, retry, random-order)
- ruff, pylint (linting)
- pre-commit (git hooks)
- sphinx (+ extensions: numpydoc, myst-parser)

## Additional Resources

- **Documentation:** https://gammasim.github.io/simtools/
- **Repository:** https://github.com/gammasim/simtools
- **Issues:** https://github.com/gammasim/simtools/issues
- **SimPipe Pipeline:** http://cta-computing.gitlab-pages.cta-observatory.org/dpps/simpipe/simpipe/latest/
- **CTAO Website:** https://www.cta-observatory.org/
- **Developer Contact:** simtools-developer@desy.de

## Quick Reference

```bash
# Setup
pip install -e '.[dev,tests]'
pre-commit install

# Testing
pytest tests/unit_tests/              # Unit tests
pytest -n 4 tests/unit_tests/         # Parallel
pytest --durations=10                 # Find slow tests

# Linting
ruff check --fix                      # Fix issues
ruff format                           # Format code
pre-commit run --all-files            # All checks

# Documentation
cd docs && make html                  # Build docs

# Git workflow
git checkout -b feature/my-feature    # New branch
# ... make changes ...
pre-commit run --all-files            # Check
git commit -m "Description"           # Commit
git push origin feature/my-feature    # Push
```

## For AI Coding Agents

When working with this codebase:

1. **Always run tests** after changes: `pytest tests/unit_tests/`
2. **Always run linters** before commits: `pre-commit run --all-files`
3. **Follow conventions** strictly (pathlib, logging, f-strings, NumPy docstrings)
4. **Write tests** for all new code (mandatory, aim for >90% coverage)
5. **Use mocking** in tests to avoid external dependencies
6. **Validate names** using `simtools.utils.names` functions
7. **Check existing code** for patterns before implementing new features
8. **Document thoroughly** with NumPy-style docstrings (70%+ coverage required)
9. **Make minimal changes** - understand the codebase first
10. **Use containers** when working with CORSIKA/sim_telarray

This is a scientific computing project with high quality standards. Code must be correct, well-tested, properly documented, and follow established conventions.
