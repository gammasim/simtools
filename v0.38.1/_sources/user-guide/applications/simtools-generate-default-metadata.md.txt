# simtools-generate-default-metadata

```{eval-rst}
.. automodule:: simtools.applications.generate_default_metadata
   :members:
   :exclude-members: main
```

```{eval-rst}
**Command line arguments**

schema_file (str, optional)
    Schema file describing the input data
    (default: simtools/schemas/metadata.metaschema.yml)
output_file (str, optional)
    Output file name.

**Example**

.. code-block:: console

    simtools-generate-default-metadata \\
        --schema_file simtools/schemas/metadata.metaschema.yml \\
        --output_file default_metadata.yml
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: generate_default_metadata
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: generate_default_metadata_to_json_file.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: generate_default_metadata_to_yml_file.yml
```
