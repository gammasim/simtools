# simtools-db-add-simulation-model-from-repository-to-db

```{eval-rst}
.. automodule:: simtools.applications.db_add_simulation_model_from_repository_to_db
   :members:
   :exclude-members: main
```

```{eval-rst}
Generates a new database with all required collections.
Follows the structure of the CTAO gitlab simulation model repository.

This is an application for DB maintainers and should not be used by the general user.

Command line arguments

input_path (str, required)
    Path of local copy of model parameter repository.
db_simulation_model (str, required)
    Name of new DB to be created.
db_simulation_model_version (str, required)
    Version of the new DB to be created.
type (str, optional)
    Type of data to be uploaded to the DB. Options are: model_parameters, production_tables.

**Examples**

Upload model data repository to the DB
Loops over all subdirectories in 'input_path' and uploads all json files to the
database (or updates an existing database with the same name):

* subdirectories starting with 'OBS' are uploaded to the 'sites' collection
* json files from the subdirectory 'configuration_sim_telarray/configuration_corsika'
  are uploaded to the 'configuration_sim_telarray/configuration_corsika' collection
* 'Files' are added to the 'files' collection
* all other json files are uploaded to collection defined in the array element description
  in 'simtools/schemas/array_elements.yml'

.. code-block:: console

    simtools-db-simulation-model-from-repository-to-db \
        --input_path /path/to/repository \
        --db_simulation_model database name \
        --db_simulation_model_version new database version \
        --type model_parameters

Upload production tables to the DB:

.. code-block:: console

    simtools-db-simulation-model-from-repository-to-db \
        --input_path /path/to/repository \
        --db_simulation_model database name \
        --db_simulation_model_version new database version \
        --type production_tables
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: db_add_simulation_model_from_repository_to_db
   :no-heading:
```
