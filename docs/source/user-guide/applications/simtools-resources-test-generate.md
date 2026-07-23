# simtools-resources-test-generate

```{eval-rst}
.. automodule:: resources_test_generate
   :members:
   :exclude-members: main
```

The application reads download and workflow configurations from the release-specific
configuration file directory in the `simtools-tests` repository.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: resources_test_generate
   :no-heading:
```

## Examples

Generate and download resources for a specific simtools release:

```console
simtools-resources-test-generate \
    --test_directory ../simtools-tests \
    --simtools_version v0.34.0 \
    --runtime_environment_file \
    ../simtools-tests/simtools-tests/v0.34.0/integration_tests/run_time.yml
```

Generate resources for a single workflow configuration file:

```console
simtools-resources-test-generate \
    --test_directory ../simtools-tests \
    --simtools_version v0.34.0 \
    --runtime_environment_file \
    ../simtools-tests/simtools-tests/v0.34.0/integration_tests/run_time.yml \
    --config_file \
    ../simtools-tests/simtools-tests/v0.34.0/integration_tests/config_files/\
    production_generate_grid_horizontal.yml
```

Test the integrity of static files in the simtools-tests repository:

```console
simtools-resources-test-generate \
    --test_directory ../simtools-tests \
    --simtools_version v0.34.0 \
    --runtime_environment_file \
    ../simtools-tests/simtools-tests/v0.34.0/integration_tests/run_time.yml \
    --test_static_files
```

Run only the download step without generating new resources:

```console
simtools-resources-test-generate \
    --test_directory ../simtools-tests \
    --simtools_version v0.34.0 \
    --runtime_environment_file \
    ../simtools-tests/simtools-tests/v0.34.0/integration_tests/run_time.yml \
    --download_only
```
