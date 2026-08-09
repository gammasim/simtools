# simtools-db-add-file-to-db

```{eval-rst}
.. automodule:: simtools.applications.db_add_file_to_db
   :members:
   :exclude-members: main
```

```{eval-rst}
The name and location of the file are required.
This application should complement the ones for updating parameters, \
adding entries to the DB and getting files from the DB.

**Command line arguments**
file_name (str or list of str, required)
    Name of the file to upload including the full path. \
    A list of files is also allowed, in which case only one -file_name is necessary, \
    i.e., python applications/db_add_file_to_db.py -file_name file_1.dat file_2.dat file_3.dat \
    If no path is given, the file is assumed to be in the CWD.
input_path (str, required if file_name is not given)
    A directory with files to upload to the DB. \
    All files in the directory with a predefined list of extensions will be uploaded.
database_name (str)
    The DB to insert the files to.

**Example**
uploading a dummy file.

.. code-block:: console

    simtools-db-add-file-to-db --file_name test_application.dat --database_name test-data

Expected final print-out message:

.. code-block:: console

    INFO::get_file_from_db(l75)::main::Got file test_application.dat from DB test-data and
    saved into .
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: db_add_file_to_db
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: db_add_file_to_db_run.yml
```
