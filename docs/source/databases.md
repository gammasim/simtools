# Databases

The simtools package uses a prototype MongoDB database to store the simulation model parameters and derived data products.
Access to the DB is handled via a dedicated API module ([db_handler](#dbhandler)).

Simulation model parameters are stored in databases (see the [Simulation Model](model_parameters.md#simulation-model) section) and synced with the [CTAO model parameter repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/model_parameters).

## Database structure

Several different databases are used in simtools.

:::{Important}
The structure of the database is currently under revisions and will change in near future.
This documentation is therefore incomplete.
:::

### Model parameters DB

The name of the model parameter database needs to be indicated by `$SIMTOOLS_DB_SIMULATION_MODEL` environmental variable and defined e.g., in the `.env` file.

Collections:

* `telescopes` with the model parameters for each telescope plus the reference design model
* `sites` with all site-specific parameters (e.g., atmosphere, magnetic field, array layouts)
* `calibration_devices` with the model parameters for e.g., the illumination devices
* `configuration_sim_telarray` with default configuration parameters for the sim_telarray simulation
* `metadata` containing tables describing the model versions
* `fs.files` with all file type entries for the model parameters (e.g., the quantum-efficiency tables)

### Derived values DB

Database with derived values DB (e.g., `Staging-CTA-Simulation-Model-Derived-Values` defined in `db_handler.DB_DERIVED_VALUES`).

Collections are:

* `derived_values` with the derived values for each telescope or site
* `fs.files` with file type derived results

### Other databases

All other currently available databases are not in use and kept for historical reasons.

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

## Browse the database

The mongoDB database can be accessed via the command-line interface `mongo` or via a GUI tool like `Robo 3T` or `Studio 3T`.

## Update the database

Model parameters should first reviewed and accepted in the [model parameter repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/model_parameters) before they are uploaded to the database (see the following sections on how to set up a local copy of the model parameter database for testing and development).

:::{Danger}
Updating the database is for experts only.
Note that updating the databases might have impact on workflows of others.
Best to define a new database when adding new values or files.
:::

The following applications are important:

* update or define a single model parameter from a json file (as defined in the model parameter repository): [add_value_from_json_to_db.py](add_value_from_json_to_db)
* upload a model parameter file: [add_file_to_db.py](add_file_to_db)
* upload all model parameters and files from the model parameter repository: [db_add_model_parameters_from_repository_to_db.py](db_add_model_parameters_from_repository_to_db)

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
