# Continuous Integration

The simtools continuous integration tests, lints, and builds code, documentation and binaries. The CI is based on GitHub Actions and defined in the `.github/workflows` directory. This page documents the simtools CI workflows.

## Units tests

Unit tests are testing the code on a module and function level, see the [unit tests](testing.md#unit-tests) documentation for more details.

Unit tests are implemented in the CI in the [CI-unittests.yml](.github/workflows/CI-unittests.yml) workflow.
Unit tests are running for different python versions and installation types (mamba, pip).
There is a scheduled run of unit tests over the main branch each night, which includes also a random-order execution of the tests (to find inter-dependencies between modules).

:::{important}
Simtools uses the CTAO sonar quality gates for code quality. Passing those gates is a requirement for merging code into the main branch.
See the [simtools sonar project page](https://sonar-ctao.zeuthen.desy.de/tutorials?id=gammasim_simtools_0d23837b-8b2d-4e54-9a98-2f1bde681f14) for the current status.
:::

## Integration tests

Integrations tests are testing the main functionality of simtools applications. Every application should have
integration tests covering the most common and important use cases. See the [integration tests](testing.md#integration-tests) documentation for more details.

The CI Integration tests is defined in the [CI-integrationtests.yml](.github/workflows/CI-integrationtests.yml) workflow.  Integration tests are running for two different model versions.

## Linting

Consistent code style is enforced by running the following linters:

- linters defined as pre-commit workflow, see [.pre-commit-config.yaml](../.pre-commit-config.yaml)
- files are checked for non-ascii characters
- linting of pyproject.toml
- linting of CITATION.cff
- running pylint on the code
- linting docker files using [hadolint](https://github.com/hadolint/hadolint)
- validate env files
- check for natural language

## Schema file validation

All schema files are validated using the `validate_file_using_schema.py` tool of simtools.
Schema CI is define in [CI-schema-validation.yml](.github/workflows/CI-schema-validation.yml).

## Documentation

Documentation is built using sphinx after every merge to main.
The documentation is built into the [CI-docs.yml](.github/workflows/CI-docs.yml) workflow.

## Binary builds

Binary builds are done in two ways:

- containers for CORSIKA and sim_telarray, see [build-docker-corsika-simtelarray-image.yml](.github/workflows/build-docker-corsika-simtelarray-image.yml)
- containers for simtools, see [build-docker-images.yml](.github/workflows/build-docker-images.yml)

Builds are done for the platforms: linux/amd64,linux/arm64/v8.

## Deployment

The Pypi deployment for each release is done using the [pypi.yml](.github/workflows/pypi.yml) workflow.
