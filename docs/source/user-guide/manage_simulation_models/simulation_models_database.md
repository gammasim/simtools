
# Simulation Models Databases

The following applications are useful for testing, maintenance or debugging the simulation model database.
Note that the remote primary production database is modified only by the simulation models gitlab CI.

Applications modifying database entries:

* update or define a single model parameter from a json file (as defined in the model parameter repository): [simtools-db-add-value-from-json-to-db](db_add_value_from_json_to_db)
* upload a model parameter file: [simtools-db-add-file-to-db](db_add_file_to_db)
* upload all model parameters and files from the model parameter repository: [simtools-db-add-simulation-model-from-repository-to-db](db_add_simulation_model_from_repository_to_db)

Applications to retrieve values or information from the database:

* list all databases and collections defined: [simtools-db-inspect-databases](db_inspect_databases)
* retrieve a single model parameter: [simtools-db-get-parameter-from-db](db_get_parameter_from_db)
* retrieve a model parameter file: [simtools-db-get-file-from-db](db_get_file_from_db)
* retrieve and print array layouts defined in the database: [simtools-db-get-array-layouts-from-db](db_get_array_layouts_from_db)

```{warning}
This section is for experts only and generally not needed for regular users.
```

## Updating the model database

```{danger}
The following sections describe functionality which might impact the simulation model database for all users.
Generally, model values should not be changed without a merge request in the [simulation models repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models.)
Tests should be done using a sandbox database before applying this to the production databases.
```

```{warning}
TODO - most below is probably application documentation?
```

### Update a single model parameter

New model parameter defined in the simtools database format (json file) can be uploaded to the database using the {ref}`simtools-add-value-from-json-to-db <db_add_value_from_json_to_db>` application.
New data files can be uploaded using {ref}`simtools-add-file-to-db <db_add_file_to_db>`.

### Upload a complete simulation model from model repository to model database

The application `db_add_model_parameters_from_repository_to_db.py` allows to export the simulation model parameters from a
model repository to the model database for a given model version.
See the [database](databases.md#databases) section for implementation details on databases in simtools.

Example:

```bash
simtools-add_model-parameters-from-repository-to-db \
      --input_path /path/to/repository \
      --db_name new_db_name
```

This application loops over all subdirectories in `input_path` and uploads all json files to the database `new_db_name` (or updates an existing database with the same name):

- subdirectories starting with `OBS` are uploaded to the `sites` collection
- json files from the subdirectory `configuration_sim_telarray/configuration_corsika` are uploaded to the `configuration_sim_telarray/configuration_corsika` collection
- `Files` are added to the `files` collection
- all other json files are uploaded to collection defined in the array element description in [simtools/schemas/array_elements.yml](https://github.com/gammasim/simtools/blob/main/simtools/schemas/array_elements.yml). Allowed values are e.g., `telescopes`, `calibration_devices`.
