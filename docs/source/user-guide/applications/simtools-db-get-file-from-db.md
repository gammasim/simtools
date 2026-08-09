# simtools-db-get-file-from-db

```{eval-rst}
.. automodule:: simtools.applications.db_get_file_from_db
   :members:
   :exclude-members: main
```

```{eval-rst}
The name of the file is required.
This application complements the ones for getting parameters, adding entries and files     to the DB.

**Command line arguments**

file_name (str or list of str, required)
    Name of the file to get including its full directory. A list of files is also allowed.
    i.e., python applications/get_file_from_db.py -file_name mirror_CTA-N-LST1_v2019-03-31.dat.
output_path (str)
    Name of the local output directory where to save the files.
    Default it $CWD.

**Example**

getting a file from the DB.

.. code-block:: console

    simtools-db-get-file-from-db --file_name mirror_CTA-N-LST1_v2019-03-31.dat

Expected final print-out message:

.. code-block:: console

    INFO::db_get_file_from_db(l82)::main::Got file mirror_CTA-N-LST1_v2019-03-31.dat from DB         CTA-Simulation-Model and saved into .
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: db_get_file_from_db
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: db_get_file_from_db_ctao-simulation-model-two-files.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: db_get_file_from_db_ctao-simulation-model.yml
```
