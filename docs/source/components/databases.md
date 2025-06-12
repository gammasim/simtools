# Databases

Simulation model parameters and production configurations are stored in databases (see the [Simulation Model](model_parameters.md#simulation-model) section) and synced with the [CTAO model repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models).
The simtools package uses a MongoDB database to store production tables simulation model parameters.

```{important}
No direct write access to the simulation model database is allowed.
Updates to the simulation models should be done via merge requests to the [CTAO model repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models).
```

For a discussion on how to use a local copy of the database for testing and development, see the [databases for developers](developer_guide/databases_for_developers.md) section.

## Simulation Models Database

The name of the model parameter database needs to be indicated by `$SIMTOOLS_DB_SIMULATION_MODEL` environmental variable and defined e.g., in the `.env` file. Use `CTAO-Simulation-ModelParameters-LATEST` to use the latest version of the CTAO simulation model database (simtools will replace `LATEST` with the latest version number).

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

## Other Databases

All other currently available databases are not in use and kept for historical reasons.

## Using the Simulation Models Database located at DESY

A simulation models database is located at DESY. Access to the database is restricted, please contact the developers in order to obtain access.

Database and access configuration is given in the `.env` file, see the [.env_template](../../.env_template) file as example:

```console
SIMTOOLS_DB_API_PORT=27017 #Port on the MongoDB server
SIMTOOLS_DB_SERVER='cta-simpipe-protodb.zeuthen.desy.de' # MongoDB server
SIMTOOLS_DB_API_USER=YOUR_USERNAME # username for MongoDB: ask the responsible person
SIMTOOLS_DB_API_PW=YOUR_PASSWORD # Password for MongoDB: ask the responsible person
SIMTOOLS_DB_API_AUTHENTICATION_DATABASE='admin'
SIMTOOLS_DB_SIMULATION_MODEL='CTAO-Simulation-ModelParameters-LATEST'
SIMTOOLS_SIMTEL_PATH='/workdir/sim_telarray'
```

## Browse the database

The mongoDB database can be accessed via the command-line interface `mongo` or via a GUI tool like `Robo 3T` or `Studio 3T`.
