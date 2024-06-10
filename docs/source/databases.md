# Databases

The simtools package uses a prototype MongoDB database for the telescopes and sites model, reference data, derived values and test data.
Access to the DB is handled via a dedicated API module. Access to the DB is restricted, please contact the developers in order to obtain access.

Simulation model parameters are stored in databases (see the [Simulation Model](model_parameters.md#simulation-model) section) and synced with the [CTAO model parameter repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/model_parameters).

Several different databases are used:

* model parameters DB (name needs to be indicated by `SIMTOOLS_DB_SIMULATION_MODEL` in your `.env` file)
* derived values DB (e.g., `Staging-CTA-Simulation-Model-Derived-Values` defined in `db_handler.DB_DERIVED_VALUES`)

:::{Important}
The structure of the database is currently under revisions and will change in near future.
This documentation is therefore incomplete.
:::

## Using the remote database located at DESY

A prototype remote database is located at DESY. Access to the database is restricted, please contact the developers in order to obtain access.

Database and access configuration is given in the `.env` file, see the [.env_template](../../.env_template) file as example:

```console
SIMTOOLS_DB_API_PORT=27017 #Port on the MongoDB server
SIMTOOLS_DB_SERVER='cta-simpipe-protodb.zeuthen.desy.de' # MongoDB server
SIMTOOLS_DB_API_USER=YOUR_USERNAME # username for MongoDB: ask the responsible person
SIMTOOLS_DB_API_PW=YOUR_PASSWORD # Password for MongoDB: ask the responsible person
SIMTOOLS_DB_API_AUTHENTICATION_DATABASE='admin'
SIMTOOLS_DB_SIMULATION_MODEL='Staging-CTA-Simulation-Model-v0-3-0'
# SIMTOOLS_DB_SIMULATION_MODEL_URL=''
SIMTOOLS_SIMTEL_PATH='/workdir/sim_telarray'
```

## Browsing the mongoDB database

The mongoDB database can be accessed via the command-line interface `mongo` or via a GUI tool like `Robo 3T` or `Studio 3T`.

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

Note that (for unknown reason) the script needs to be executed twice.

### Filling the local database instance

#### Option 1: Fill local database from remote DB dump

Access to a database dump of the production database is required. It is assumed that the dumps
are located in the directory `database_scripts/dumps`.

The script `./dump_remote_db.sh` can be used to create a dump of the production database (requires access to this DB and the `mongodump` tool).

Use then the `upload_dump_to_local_db.sh` to upload this dump to the local database instance.

Note that database names are hardcoded in the scripts and need to be adjusted accordingly.

#### Option 2: Fill local database from model parameter repository

The script `upload_from_model_repository_to_local_db.sh` uses the [model parameter repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/model_parameters) from the CTAO gitlab and
uploads its contents to the local database instance.

Note that database names are hardcoded in the scripts and need to be adjusted accordingly.

## Purge the local database instance and all networks, images, containers

The script `purge_local_db.sh` stops and removes the container and deletes all networks, images, and containers.

:::{Danger}
Attention: this script removes all local docker containers, images, and networks without awaiting confirmation.
:::

## Using the local database instance

This requires the following changes to the settings of the environmental variables in `.evn`:

```console
# Environmental variables
SIMTOOLS_DB_API_PORT=27017 #Port on the MongoDB server
SIMTOOLS_DB_SERVER='localhost'
SIMTOOLS_DB_API_USER='api' # username for MongoDB
SIMTOOLS_DB_API_PW='password' # Password for MongoDB
SIMTOOLS_DB_API_AUTHENTICATION_DATABASE='admin'
SIMTOOLS_DB_SIMULATION_MODEL='Staging-CTA-Simulation-Model-v0-3-0'
```

`SIMTOOLS_DB_SIMULATION_MODEL` is set as an example here to `Staging-CTA-Simulation-Model-v0-3-0` and should be changed accordingly.

For using simtools inside a container:

* set the `SIMTOOLS_DB_SERVER` in the `.env` file to SIMTOOLS_DB_SERVER='simtools-mongodb'.
* connect to the local network adding `--network simtools-mongo-network` to the `docker/podman run` command, e.g, `podman run --rm -it -v "$(pwd)/:/workdir/external" --network simtools-mongo-network ghcr.io/gammasim/simtools-dev:latest bash`
