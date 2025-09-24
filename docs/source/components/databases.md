# Databases

Simulation model parameters and production configurations are stored in databases (see the [Simulation Model](model_parameters.md#simulation-model) section) and synced with the [CTAO model repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models).
The simtools package uses a MongoDB database to store production tables and simulation model parameters.

```{important}
No direct write access to the simulation model database is allowed.
Updates to the simulation models should be done via merge requests to the [CTAO model repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models).
```

## Simulation Models Database

The name and version of the model parameter database needs to be indicated by `$SIMTOOLS_DB_SIMULATION_MODEL` and `$DB_SIMULATION_MODEL_VERSION` environmental variables and defined e.g., in the `.env` file.

Collections:

* `telescopes` with the model parameters for each telescope plus the reference design model
* `sites` with all site-specific parameters (e.g., atmosphere, magnetic field, array layouts)
* `calibration_devices` with the model parameters for e.g., the illumination devices
* `configuration_sim_telarray` with configuration parameters for the sim_telarray simulation
* `configuration_corsika` with configuration parameters for the CORSIKA simulation
* `metadata` containing tables describing the model versions
* `fs.files` with all file type entries for the model parameters (e.g., the quantum-efficiency tables)

### Update the Simulation Models Database

Model parameters should first be reviewed and accepted in the [model repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models) before they are uploaded to the database.
To update the database, release a new version of the simulation models repository and the CI will automatically update the database with the new model parameters.

## Using the Simulation Models Database located at DESY

A simulation models database is located at DESY. Access to the database is restricted, please contact the developers in order to obtain access.

Database and access configuration is given in the `.env` file, see the [.env_template](../../.env_template) file as example:

```console
SIMTOOLS_DB_API_PORT=27017 # Port on the database server
SIMTOOLS_DB_SERVER='cta-simpipe-protodb.zeuthen.desy.de' # MongodatabaseDB server
SIMTOOLS_DB_API_USER=YOUR_USERNAME # username for database: ask the responsible person
SIMTOOLS_DB_API_PW=YOUR_PASSWORD # Password for database: ask the responsible person
SIMTOOLS_DB_API_AUTHENTICATION_DATABASE='admin'
SIMTOOLS_DB_SIMULATION_MODEL_VERSION='v0.9.0' # Version of the simulation model database (adjust accordingly)
SIMTOOLS_DB_SIMULATION_MODEL='CTAO-Simulation-Model'
SIMTOOLS_SIMTEL_PATH='/workdir/sim_telarray'
```

## Browse the database

The MongoDB database can be accessed via the command-line interface `mongo` or via a GUI tool like `Robo 3T` or `Studio 3T`.

## Setup a local copy of the model parameter database

The production version of model-parameter database is a MongoDB instance running on a server at DESY.
For testing and development, it is recommend to work with a local copy of the database.
The following scripts allow to setup and fill a local database running in a container.

All scripts to setup and fill a local database instance are located in the [database_scripts](../../database_scripts/) directory.

```{warning}
Be careful of the settings described below to avoid accidental overwriting of the remote database.
```

### Startup and configure local database instance

The script [setup_local_db.sh](../../database_scripts/setup_local_db.sh) generates a local database instance in a container:

* downloads a MongoDB Docker image
* starts a container with the image and initialize a new database
* add a user with `readWrite` role
* defines a container network called `simtools-mongo-network` (check with `podman network ls`)

### Fill the local database instance

#### Option 1 (preferred): Fill local database from model parameter repository

The script `upload_from_model_repository_to_db.sh` uses the [model repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models) from the CTAO GitLab and
uploads its contents to the local database instance.

#### Option 2: Fill local database from remote DB dump

Access to a database dump of the production database is required. It is assumed that the dumps
are located in the directory `database_scripts/dumps`.

The script `./dump_remote_db.sh` can be used to create a dump of the production database (requires access to this DB and the `mongodump` tool).

Use then the `upload_dump_to_local_db.sh` to upload this dump to the local database instance. Take care to use the correct environments for dumping and uploading.

Note that database names are hardcoded in the scripts and need to be adjusted accordingly.

### Upload to and download from the local database instance

Startup a container and connect to the local network adding `--network simtools-mongo-network` to the `docker/podman run` command, e.g,:

```bash
podman run --rm -it -v "$(pwd)/:/workdir/external" --network simtools-mongo-network ghcr.io/gammasim/simtools-dev:latest bash
```

Modify the environment file `.env` to be used with a container:

```console
# Environment variables
SIMTOOLS_DB_API_PORT=27017 # Port on the database server
SIMTOOLS_DB_SERVER='simtools-MongoDB'
SIMTOOLS_DB_API_USER='api' # username for database
SIMTOOLS_DB_API_PW='password' # Password for database
SIMTOOLS_DB_API_AUTHENTICATION_DATABASE='admin'
SIMTOOLS_DB_SIMULATION_MODEL_VERSION='v0.9.0' # Version of the simulation model database (adjust accordingly)
SIMTOOLS_DB_SIMULATION_MODEL='CTAO-Simulation-Model'
SIMTOOLS_SIMTEL_PATH='/workdir/sim_telarray'
```

Fill the model parameter database from the model repository (parameters must match the version defined in the `.env` file):

```console
cd /workdir/external/database_scripts
./upload_from_model_repository_to_db.sh CTAO-Simulation-Model v20.9.0
```

### Purge the local database instance

The script `purge_local_db.sh` stops and removes the container and deletes all networks, images, and containers.

:::{Danger}
Attention: this script removes all local Docker containers, images, and networks without awaiting confirmation.
:::
