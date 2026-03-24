# Databases

Simulation model parameters and production configurations are stored in databases (see the [Simulation Model](model_parameters.md#simulation-model) section) and synced with the [CTAO model repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models).
The simtools package uses a MongoDB database to store production tables and simulation model parameters.

```{important}
No direct write access to the simulation model database is allowed to general users.
Updates to the simulation models should be done via merge requests to the [CTAO model repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models).
New releases of the model repository will automatically trigger an update of the database with the new model parameters.
```

## Simulation Models Database

The name and version of the model parameter database need to be indicated by the
`$SIMTOOLS_DB_SIMULATION_MODEL` and `$SIMTOOLS_DB_SIMULATION_MODEL_VERSION`
environment variables and defined, e.g., in the `.env` file.

Collections:

* `telescopes` with the model parameters for each telescope plus the reference design model
* `sites` with all site-specific parameters (e.g., atmosphere, magnetic field, array layouts)
* `calibration_devices` with the model parameters for e.g., the illumination devices
* `configuration_sim_telarray` with configuration parameters for the sim_telarray simulation
* `configuration_corsika` with configuration parameters for the CORSIKA simulation
* `metadata` containing tables describing the model versions
* `fs.files` with all file type entries for the model parameters (e.g., the quantum-efficiency tables)

### Browse the database

The MongoDB database can be accessed via the `mongosh` command-line interface (or the legacy `mongo` CLI) or via a GUI tool like `Robo 3T` or `Studio 3T`.

### Update the Simulation Models Database

Model parameters should first be reviewed and accepted in the [model repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models) before they are uploaded to the database.
To update the database, release a new version of the simulation models repository and the CI will automatically update the database with the new model parameters.

## Using the Simulation Models Database located at DESY

A simulation models database is located at DESY. Access to the database is restricted, please contact the developers in order to obtain access.

Database and access configuration is given in the `.env` file, see the [.env_template](../../.env_template) file as example:

```console
SIMTOOLS_DB_API_PORT=27017 # Port on the database server
SIMTOOLS_DB_SERVER='cta-simpipe-protodb.zeuthen.desy.de' # MongoDB server
SIMTOOLS_DB_API_USER=YOUR_USERNAME # username for database: ask the responsible person
SIMTOOLS_DB_API_PW=YOUR_PASSWORD # Password for database: ask the responsible person
SIMTOOLS_DB_API_AUTHENTICATION_DATABASE='admin'
SIMTOOLS_DB_SIMULATION_MODEL_VERSION='v0.9.0' # Version of the simulation model database (adjust accordingly)
SIMTOOLS_DB_SIMULATION_MODEL='CTAO-Simulation-Model'
```

## Setup a local copy of the model parameter database

The production version of model-parameter database is a MongoDB instance running on a server at DESY.
For testing and development, it is recommended to work with a local copy of the database.
The following scripts allow setting up and filling a local database running in a container.

Scripts to set up a local database instance are located in the [database_scripts](../../database_scripts/) directory.

```{warning}
Be careful of the settings described below to avoid accidental overwriting of the remote database.
```

### Startup and configure local database instance

The script [setup_local_db.sh](../../database_scripts/setup_local_db.sh) generates a local database instance in a container:

```bash
cd database_scripts
./setup_local_db.sh
```

This script:

* downloads a MongoDB container image
* starts a container with the image and initializes a new database
* creates an API user with read/write access
* defines a container network called `simtools-mongo-network` (check with `podman network ls`)

### Fill the local database instance

#### Option 1 (preferred): Fill local database from model parameter repository

The simtools package includes the application `simtools-db-upload-model-repository` to upload the model parameters from a
local clone of the [model repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models) to a local database instance.
The repository is cloned by the application to a temporary directory (`--tmp_dir`).

To upload the model parameters, follow these steps:

1. Start a container and connect to the local network by adding `--network simtools-mongo-network` to the `podman run` command, e.g.,

```bash
podman run --rm -it -v "$(pwd)/:/workdir/external" --network simtools-mongo-network ghcr.io/gammasim/simtools-dev:latest bash
```

2. Modify the environment file `.env` to set the simulation model version. Best practice is to use a released version of the model repository, e.g.,

```console
# Version of the simulation model database (adjust accordingly)
SIMTOOLS_DB_SIMULATION_MODEL_VERSION=v0.12.0
SIMTOOLS_DB_SIMULATION_MODEL=CTAO-Simulation-Model

# Local MongoDB connection created by setup_local_db.sh
SIMTOOLS_DB_SERVER=simtools-mongodb
SIMTOOLS_DB_API_USER=api
SIMTOOLS_DB_API_PW=password
SIMTOOLS_DB_API_PORT=27017
SIMTOOLS_DB_API_AUTHENTICATION_DATABASE=admin
```

3. Fill the model parameter database from the model repository (parameters must match the version defined in the `.env` file):

```console
simtools-db-upload-model-repository --db_simulation_model_version v0.12.0
```

For development purposes, a specific branch of the model repository can be used by providing the `--branch` argument to the upload application.

#### Option 2: Fill local database from remote DB dump

Access to a database dump of the production database is required. It is assumed that the dumps
are located in the directory `database_scripts/dump`.

The script `./dump_remote_db.sh` can be used to create a dump of the production database (requires access to this DB and the `mongodump` tool).

Then use `upload_dump_to_local_db.sh` to upload this dump to the local database instance. Take care to use the correct environment variables for dumping and uploading.

Note that database names are hardcoded in the scripts and need to be adjusted accordingly.

### Purge the local database instance

The script `purge_local_db.sh` stops and removes the container and deletes all networks, images, and containers.

```{danger}
Attention: this script removes all local containers, images, and networks without asking for confirmation.
```
