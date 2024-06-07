# Collection of database scripts

This directory contains a collection of scripts that can be used to interact with the database:

* dump or copy databases
* generate a local copy of the model parameter database.

## Running a local copy of the model parameter database

The model parameter database is a mongoDB instance running on a server at DESY.
For testing and development, it might be useful to work with a local copy of the database.

**The following steps are "experimental" and need further testing.**

### Startup and configure local data base instance

The scripts `setup_local_db.sh` generates a local database instance in a container:

* downloads a mongoDB docker image
* starts a container with the image and initialize a new database
* add a user 'api' with 'readWrite' role

Note that for unknown reason, the script needs to be executed twice (!!), in case error messages are shown during the first run.

### Fill local database from remote DB dump

Access to a database dump of the production database is required. It is assumed that the dumps
are located in the directory `database_scripts/dumps`.

The script `./dump_remote_db.sh` can be used to create a dump of the production database (requires access to this DB and the `mongodump` tool).

Use then the `upload_dump_to_local_db.sh` to upload this dump to the local database instance.

Note that database names are hardcoded in the scripts and need to be adjusted accordingly.

### Fill local database from model parameter repository

The script `upload_from_model_repository_to_local_db.sh` uses the model parameter repository from the CTAO gitlab and
uploads its contents to the local database instance.

Note that database names are hardcoded in the scripts and need to be adjusted accordingly.

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

## Purge the local database instance and all networks, images, containers

The script `purge_local_db.sh` stops and removes the container and deletes all networks, images, and containers.

**Attention: this script removes all local docker containers, images, and networks without awaiting confirmation.**
