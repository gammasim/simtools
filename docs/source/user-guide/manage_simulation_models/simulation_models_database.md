
# Simulation Models Databases

The following applications are useful for testing, maintenance or debugging the simulation model database.
Note that the remote primary production database is modified only by the simulation models gitlab CI.

## Inspecting the model database

Applications to retrieve values or information from the database:

* list all databases and collections defined: [simtools-db-inspect-databases](db_inspect_databases)
* retrieve a single model parameter: [simtools-db-get-parameter-from-db](db_get_parameter_from_db)
* retrieve a model parameter file: [simtools-db-get-file-from-db](db_get_file_from_db)
* retrieve and print array layouts defined in the database: [simtools-db-get-array-layouts-from-db](db_get_array_layouts_from_db)

## Updating the model database

```{danger}
The following sections describe functionality which might impact the simulation model database for all users.
Generally, model values should not be changed without a merge request in the [simulation models repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models.)
Tests should be done using a sandbox database before applying this to the production databases.
```

Applications modifying database entries:

* update or define a single model parameter from a json file (as defined in the model parameter repository): [simtools-db-add-value-from-json-to-db](db_add_value_from_json_to_db)
* upload a model parameter file: [simtools-db-add-file-to-db](db_add_file_to_db)
* upload all model parameters and files from the model parameter repository: [simtools-db-add-simulation-model-from-repository-to-db](db_add_simulation_model_from_repository_to_db)
