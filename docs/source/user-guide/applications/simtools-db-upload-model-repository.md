# simtools-db-upload-model-repository

```{eval-rst}
.. automodule:: simtools.applications.db_upload_model_repository
   :members:
   :exclude-members: main
```

## Overview

This application uploads a CTAO simulation-model repository to MongoDB. It uploads model
parameters and production tables under the selected simulation-model name and version.

The application can clone the repository using a version tag or branch, or use an existing local
checkout with `repository_dir`. It retries repository operations up to `max_attempts` times.

Review the target database carefully before running an upload. Remote database uploads may require
confirmation and should use credentials from the configured environment file.

## Repository selection

Use `repository_dir` for a local checkout. If it is omitted, the application clones the default
simulation-model repository into `tmp_dir`. `branch` selects a branch; otherwise the model version
is used as the repository tag.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: db_upload_model_repository
   :no-heading:
```

## Examples

Upload a released repository version (includes cloning the simulations-models repository):

```console
simtools-db-upload-model-repository \
    --db_simulation_model CTAO-Simulation-Model \
    --db_simulation_model_version v0.16.0
```

Upload an existing local checkout:

```console
simtools-db-upload-model-repository \
    --db_simulation_model CTAO-Simulation-Model \
    --db_simulation_model_version v0.16.0 \
    --repository_dir /path/to/simulation-models
```
