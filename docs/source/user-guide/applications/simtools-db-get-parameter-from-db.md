# simtools-db-get-parameter-from-db

```{eval-rst}
.. automodule:: simtools.applications.db_get_parameter_from_db
   :members:
   :exclude-members: main
```

```{eval-rst}
The application supports three output modes:

1. Print the database entry to stdout.
2. Write the database entry to a JSON or YAML file using output_file.
3. Export table-type model parameters using export_model_file.

The export_model_file mode is type-dependent:

- File-backed parameters are exported as model files.
    Use output_file to override the exported file name.
- Dict-backed table parameters are exported as ECSV, using output_file as the base name.

For file-backed parameters, export_model_file_as_table can be added to also write an
ECSV representation next to the exported file.

**Command line arguments**

parameter (str, required)
    Parameter name

parameter_version (str, optional)
    Parameter version

model_version (str, required)
    Model version

site (str, required)
    South or North.

telescope (str, optional)
    Telescope model name (e.g. LST-1, SST-D, ...)

output_file (str, optional)
    Output file name for writing the database entry, overriding the exported
    file name for file-backed parameters, or base file name for exporting
    dict-backed tables as ECSV.

export_model_file (bool, optional)
    Export parameter data (model files for file-backed parameters, ECSV for
    dict-backed table parameters).

export_model_file_as_table (bool, optional)
    Export file-backed parameters as astropy tables in addition to the
    original file export. Use together with export_model_file.

**Raises**

KeyError in case the parameter requested does not exist in the model parameters.

**Example**

Print the mirror_list parameter entry used for a given model_version.

.. code-block:: console

    simtools-db-get-parameter-from-db --parameter mirror_list \\
            --site North --telescope LSTN-01 \\
            --model_version 5.0.0

Write the database entry for a parameter to a JSON file.

.. code-block:: console

    simtools-db-get-parameter-from-db --parameter array_element_position_ground \\
            --site North --telescope LSTN-01 \\
            --parameter_version 6.0.0 \\
            --output_file array_element_position_ground.json

Export a file-backed parameter using the original file name stored in the database.

.. code-block:: console

    simtools-db-get-parameter-from-db --parameter mirror_list \\
            --site North --telescope LSTN-01 \\
            --parameter_version 1.0.0 \\
            --export_model_file

Export a file-backed parameter and override the output file name.

.. code-block:: console

    simtools-db-get-parameter-from-db --parameter mirror_list \
            --site North --telescope LSTN-01 \
            --parameter_version 1.0.0 \
            --export_model_file --output_file my_mirror_list.dat

Export a file-backed parameter and also write an ECSV table representation.

.. code-block:: console

    simtools-db-get-parameter-from-db --parameter mirror_reflectivity \\
            --site North --telescope LSTN-01 \\
            --model_version 6.0.2 \\
            --export_model_file --export_model_file_as_table

Export a dict-backed table parameter as ECSV. The .ecsv suffix is added automatically.

.. code-block:: console

    simtools-db-get-parameter-from-db --parameter fadc_pulse_shape \\
            --site North --telescope LSTN-01 \\
            --parameter_version 2.0.0 \\
            --export_model_file --output_file fadc_pulse_shape
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: db_get_parameter_from_db
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: db_get_parameter_from_db_array_element_position_ground.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: db_get_parameter_from_db_site_parameter.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: db_get_parameter_from_db_telescope_model_version.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: db_get_parameter_from_db_telescope_parameter_version.yml
```
