# simtools-submit-data-from-external

```{eval-rst}
.. automodule:: simtools.applications.submit_data_from_external
   :members:
   :exclude-members: main
```

```{eval-rst}
Input data and metadata is validated, and if necessary enriched
and converted following a pre-described schema.

**Command line arguments**

input_meta (str, optional)
    input meta data file (yml format)
input_data_file (str, optional)
    input data file
schema_file (str, optional)
    Schema describing the input data

**Example**

Submit mirror measurements with associated metadata:

.. code-block:: console

    simtools-submit-data-from-external \\
        --input_meta ./tests/resources/MLTdata-preproduction.meta.yml \\
        --input_data_file ./tests/resources/MLTdata-preproduction.ecsv \\
        --schema_file src/simtools/schemas/input/MST_mirror_2f_measurements.schema.yml \\
        --output_file TEST-submit_data_from_external.ecsv

Expected final print-out message:

.. code-block:: console

    INFO::model_data_writer(l70)::write_data::Writing data to \\
        /simtools/simtools-output/d-2023-07-31/TEST-submit_data_from_external.ecsv
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: submit_data_from_external
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: submit_data_from_external_submit_table.yml
```
