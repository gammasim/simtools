# simtools-db-add-value-from-json-to-db

```{eval-rst}
.. automodule:: simtools.applications.db_add_value_from_json_to_db
   :members:
   :exclude-members: main
```

```{eval-rst}
**Command line arguments**

file_name (str, required)
    Name of the file to upload including the full path.
db_collection (str, required)
    The DB collection to which to add the file.
db (str)
    The DB to insert the files to.

**Example**

Upload a file to sites collection:

.. code-block:: console

    simtools-add-value-from-json-to-db \\
        --file_name new_value.json --db_collection sites
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: db_add_value_from_json_to_db
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: db_add_value_from_json_to_db_run.yml
```
