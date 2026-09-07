# simtools-db-upload-model-repository

```{eval-rst}
.. automodule:: simtools.applications.db_upload_model_repository
   :members:
   :exclude-members: main
```

## Overview

This application uploads a CTAO simulation-model repository to MongoDB. It uploads model
parameters and production tables under the selected simulation-model name and release tag.

The application can clone the repository using a release tag or branch, or use an existing local
checkout with `repository_dir`. It retries repository operations up to `max_attempts` times.

Review the target database carefully before running an upload. Remote database uploads may require
confirmation and should use credentials from the configured environment file.

## Repository selection

Use `repository_dir` for a local checkout. If it is omitted, the application clones the default
simulation-model repository into `tmp_dir`. `branch` selects a branch; otherwise the selected
release tag is used as the repository tag.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: db_upload_model_repository
   :no-heading:
```

## Examples

Upload a released repository tag (includes cloning the simulation-models repository):

```console
simtools-db-upload-model-repository \
    --db_simulation_model CTAO-Simulation-Model \
    --db_simulation_model_tag v0.16.0
```

Upload an existing local checkout:

```console
simtools-db-upload-model-repository \
    --db_simulation_model CTAO-Simulation-Model \
    --db_simulation_model_tag v0.16.0 \
    --repository_dir /path/to/simulation-models
```

```{eval-rst}
.. simtools-integration-example::
    :file: db_upload_model_repository_released_repository.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: db_upload_model_repository_local_repository.yml
```
