# simtools-export-sim-telarray-metadata-schema

```{eval-rst}
.. automodule:: simtools.applications.export_sim_telarray_metadata_schema
   :members:
   :exclude-members: main
```

```{eval-rst}
The exported schema combines generated metadata definitions from
``sim_telarray_meta_parameters.schema.yml`` with metadata derived from
model-parameter schemas.

**Command line arguments**
output_file (str)
    Output file name.
source_type (str, optional)
    Export all metadata, only generated metadata, or only model-parameter-derived metadata.
schema_version (str, optional)
    Registry schema version.

**Example**
.. code-block:: console

    simtools-export-sim-telarray-metadata-schema --output_file metadata.yml
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: export_sim_telarray_metadata_schema
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: export_sim_telarray_metadata_schema_json.yml
```
