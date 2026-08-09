# simtools-plot-tabular-data-for-model-parameter

```{eval-rst}
.. automodule:: simtools.applications.plot_tabular_data_for_model_parameter
   :members:
   :exclude-members: main
```

```{eval-rst}
Uses plotting configurations as defined in the model parameters schema files.

**Command line arguments**
parameter (str, required)
    Model parameter to plot (e.g., 'atmospheric_profile').
parameter_version (str, required)
    Version of the model parameter to plot (e.g., '1.0.0').
site (str, required)
    Site for which the model parameter is defined (e.g., 'North').
telescope (str, optional)
    Telescope for which the model parameter is defined (e.g., 'LSTN-01').
output_file (str, required)
    Output file name (without suffix).
plot_type (str, optional)
    Type of plot as defined in the schema file.
    Use '--plot_type all' to plot all types defined in the schema.

**Example**

Plot tabular data for a specific type defined in the schema file:

.. code-block:: console

    simtools-plot-tabular-data-for-model-parameter \\
        --parameter atmospheric_profile \\
        --parameter_version 1.0.0 \\
        --site North \\
        --plot_type refractive_index_vs_altitude

Plot tabular data for all types defined in the schema file:

.. code-block:: console

    simtools-plot-tabular-data-for-model-parameter \\
        --parameter fadc_pulse_shape
        --parameter_version 1.0.0 \\
        --site North \\
        --telescope LSTN-01 \\
        --plot_type all
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: plot_tabular_data_for_model_parameter
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: plot_tabular_data_for_model_parameter_atmospheric_density_all.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: plot_tabular_data_for_model_parameter_atmospheric_density_one_type.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: plot_tabular_data_for_model_parameter_fadc_pulse_shape.yml
```
