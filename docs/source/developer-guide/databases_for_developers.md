# Databases for Developers

```{important}
No direct write access to the simulation model database is allowed for developers (with the exception of maintainers).
Updates to the simulation models databases should be done via merge requests to the [CTAO model repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models) and subsequent releases (the gitlab CI will automatically update the database with the new model parameters).
```


The following applications are important:

* update or define a single model parameter from a json file (as defined in the model parameter repository): [db_add_value_from_json_to_db.py](db_add_value_from_json_to_db)
* upload a model parameter file: [db_add_file_to_db.py](db_add_file_to_db)
* upload all model parameters and files from the model parameter repository: [db_add_simulation_model_from_repository_to_db.py](db_add_simulation_model_from_repository_to_db)

## Configure and use a local copy of the model parameter database

The production version of model-parameter database is a mongoDB instance running on a server at DESY.
For testing and development, it might be useful to work with a local copy of the database.
The following scripts allow to setup and fill a local database running in a container.

All scripts to setup and fill a local database instance are located in the [database_scripts](../../database_scripts/) directory.

### Startup and configure local database instance

The script [setup_local_db.sh](../../database_scripts/setup_local_db.sh) generates a local database instance in a container:

* downloads a mongoDB docker image
* starts a container with the image and initialize a new database
* add a user with 'readWrite' role
* defines a container network called `simtools-mongo-network` (check with `podman network ls`)

### Fill the local database instance

#### Option 1: Fill local database from remote DB dump

Access to a database dump of the production database is required. It is assumed that the dumps
are located in the directory `database_scripts/dumps`.

The script `./dump_remote_db.sh` can be used to create a dump of the production database (requires access to this DB and the `mongodump` tool).

Use then the `upload_dump_to_local_db.sh` to upload this dump to the local database instance.

Note that database names are hardcoded in the scripts and need to be adjusted accordingly.

#### Option 2: Fill local database from model parameter repository

The script `upload_from_model_repository_to_db.sh` uses the [model repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models) from the CTAO gitlab and
uploads its contents to the local database instance.

Note that repository branches are hardcoded in the scripts and need to be adjusted accordingly.

**If you get timeout errors, try to run it run from within the simtools container**

### Purge the local database instance and all networks, images, containers

The script `purge_local_db.sh` stops and removes the container and deletes all networks, images, and containers.

:::{Danger}
Attention: this script removes all local docker containers, images, and networks without awaiting confirmation.
:::

### Use the local database instance (with and without docker)

This requires the following changes to the settings of the environmental variables in `.evn`:

```console
# Environmental variables
SIMTOOLS_DB_API_PORT=27017 #Port on the MongoDB server
SIMTOOLS_DB_SERVER='localhost'
SIMTOOLS_DB_API_USER='api' # username for MongoDB
SIMTOOLS_DB_API_PW='password' # Password for MongoDB
SIMTOOLS_DB_API_AUTHENTICATION_DATABASE='admin'
SIMTOOLS_DB_SIMULATION_MODEL='STAGING-CTA-Simulation-Model-LATEST'
```

`SIMTOOLS_DB_SIMULATION_MODEL` is set as an example here to `STAGING-CTAO-Simulation-ModelParameters-LATEST` and should be changed accordingly.

For using simtools inside a container:

* set the `SIMTOOLS_DB_SERVER` in the `.env` file to SIMTOOLS_DB_SERVER='simtools-mongodb'.
* connect to the local network adding `--network simtools-mongo-network` to the `docker/podman run` command, e.g,:

```bash
podman run --rm -it -v "$(pwd)/:/workdir/external" --network simtools-mongo-network ghcr.io/gammasim/simtools-dev:latest bash
```

For completeness, here the full `.env` file to be used with a container:

```console
# Environmental variables
SIMTOOLS_DB_API_PORT=27017 #Port on the MongoDB server
SIMTOOLS_DB_SERVER='simtools-mongodb'
SIMTOOLS_DB_API_USER='api' # username for MongoDB
SIMTOOLS_DB_API_PW='password' # Password for MongoDB
SIMTOOLS_DB_API_AUTHENTICATION_DATABASE='admin'
SIMTOOLS_DB_SIMULATION_MODEL='CTAO-Simulation-ModelParameters-LATEST'
SIMTOOLS_SIMTEL_PATH='/workdir/sim_telarray'
```
