# Continuous Integration

The simtools continuous integration tests, lints, and builds code, documentation and binaries. The CI is based on GitHub Actions and defined in the `.github/workflows` directory.

## CI Units and Integration tests

Tests are implemented in the CI as:

- [unit tests](testing.md#unit-tests) in the [CI-unittests.yml](.github/workflows/CI-unittests.yml) workflow.
- [integration tests](testing.md#integration-tests) in the [CI-integrationtests.yml](.github/workflows/CI-integrationtests.yml) workflow.

Unit tests are running for different python versions  and installation types (mamba, pip).
There is a scheduled run of unit tests over the main branch each night, which includes also a random-order execution of the tests (to find inter-dependencies between modules).

:::{important}
Simtools uses the CTAO sonar quality gates for code quality. Passing those gates is a requirement for merging code into the main branch.
See the [simtools sonar project page](https://sonar-cta-dpps.zeuthen.desy.de/dashboard?id=gammasim_simtools_AY_ssha9WiFxsX-2oy_w) for the current status.
:::

Integration tests are running for two different model versions.

## CI Linting

Consistent code style is enforced by running the following linters:

- linters defined as pre-commit workflow, see [.pre-commit-config.yaml](../.pre-commit-config.yaml)
- files are checked for non-ascii characters
- linting of pyproject.toml
- linting of CITATION.cff
- running pylint on the code
- linting docker files using [hadolint](https://github.com/hadolint/hadolint)
- validate env files
- check for natural language

## CI Schema files

All schema files are validated using the `validate_file_using_schema.py` tool of simtools.
Schema CI is define in [CI-schema-validation.yml](.github/workflows/CI-schema-validation.yml).

## CI Documentation

Documentation is built using sphinx after every merge to main.
The documentation is built in the [CI-docs.yml](.github/workflows/CI-docs.yml) workflow.

## CI Binary builds

Binary builds are done in two ways:

- containers for CORSIKA and sim_telarray, see [build-docker-corsika-simtelarray-image.yml](.github/workflows/build-docker-corsika-simtelarray-image.yml)
- containers for simtools, see [build-docker-images.yml](.github/workflows/build-docker-images.yml

Builds are done for the platforms: linux/amd64,linux/arm64/v8.
