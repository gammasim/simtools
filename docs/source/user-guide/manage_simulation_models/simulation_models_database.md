
# Simulation Models Database and Repository

The following applications are for testing, maintenance or debugging the simulation model database and repository.
Note that the remote primary production database is modified only by the simulation models GitLab CI.

## Simulation model database

### Inspecting the model database

Applications to retrieve values or information from the database:

* list all databases and collections defined: [simtools-db-inspect-databases](db_inspect_databases)
* retrieve a single model parameter: [simtools-db-get-parameter-from-db](db_get_parameter_from_db)
* retrieve a model parameter file: [simtools-db-get-file-from-db](db_get_file_from_db)
* retrieve and print array layouts defined in the database: [simtools-db-get-array-layouts-from-db](db_get_array_layouts_from_db)

### Updating the model database

```{danger}
The following sections describe functionality which might impact the simulation model database for all users.
Generally, model values should not be changed without a merge request in the [simulation models repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models.)
Tests should be done using a sandbox database before applying this to the production databases.
```

Applications modifying database entries:

* update or define a single model parameter from a JSON file (as defined in the model parameter repository): [simtools-db-add-value-from-json-to-db](db_add_value_from_json_to_db)
* upload a model parameter file: [simtools-db-add-file-to-db](db_add_file_to_db)
* upload all model parameters and files from the model parameter repository: [simtools-db-add-simulation-model-from-repository-to-db](db_add_simulation_model_from_repository_to_db)

## Simulation model repository

Applications to manage the simulation model repository:

* verify consistency and completeness of production tables and model parameters: [simtools-maintain-simulation-model-verify-production-tables](maintain_simulation_model_verify_production_tables)
* generate a new simulation model production tables by copying existing table and apply modifications: [simtools-maintain-simulation-model-add-production-table](maintain_simulation_model_add_production_table)
* compare two simulation model production directories and report differences: [simtools-maintain-simulation-model-compare-productions](maintain_simulation_model_compare_productions)
