# simtools-submit-model-parameter-from-external

```{eval-rst}
.. automodule:: simtools.applications.submit_model_parameter_from_external
   :members:
   :exclude-members: main
```

```{eval-rst}
Input and metadata is validated, and if necessary enriched and converted following
the model parameter schemas. Model parameter data is written in the simtools-style
json format, metadata as a yaml file.

**Command line arguments**
parameter (str)
    model parameter name
value (str, value)
    input value (number, string, string-type lists)
instrument (str)
    instrument name.
site (str)
    site location.
parameter_version (str)
    Parameter version.
model_parameter_schema_version (str, optional)
    Version of the model-parameter schema to use for validation and value interpretation.
input_meta (str, optional)
    input meta data file (yml format)

**Example**

Submit the number of gains for the LSTN-design readout chain:

.. code-block:: console

    simtools-submit-model-parameter-from-external \
        --parameter num_gains \\
        --value 2 \\
        --instrument LSTN-design \\
        --site North \\
        --parameter_version 0.1.0 \\
        --input_meta num_gains.metadata.yml
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: submit_model_parameter_from_external
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: submit_model_parameter_from_external_submit_asum_threshold.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: submit_model_parameter_from_external_submit_focus_offset.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: submit_model_parameter_from_external_submit_mirror_list.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: submit_model_parameter_from_external_submit_num_gains_wildcards.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: submit_model_parameter_from_external_submit_reference_point_altitude.yml
```
