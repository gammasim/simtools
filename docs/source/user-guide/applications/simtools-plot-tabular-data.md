# simtools-plot-tabular-data

```{eval-rst}
.. automodule:: simtools.applications.plot_tabular_data
   :members:
   :exclude-members: main
```

```{eval-rst}
Plots tabular data in one of two modes.

- Data file mode (``--plot_config``): uses a plotting configuration file to define the data to be
  plotted and all plotting details.
- Model parameter mode (``--parameter``): uses the plotting configurations defined in the model
  parameter schema files. Use ``--plot_type all`` to plot all types defined in the schema.

**Command line arguments**

plot_config (str, required in data file mode)
    Plotting configuration file name.
parameter (str, required in model parameter mode)
    Model parameter to plot (e.g., 'atmospheric_profile').
output_file (str, required in data file mode)
    Output file name (without suffix).
parameter_version (str, required in model parameter mode)
    Version of the model parameter to plot (e.g., '1.0.0').
site (str, required in model parameter mode)
    Site for which the model parameter is defined (e.g., 'North').
telescope (str, optional)
    Telescope for which the model parameter is defined (e.g., 'LSTN-01').
plot_type (str, required in model parameter mode)
    Type of plot as defined in the schema file.

**Example**

Plot tabular data using a configuration file.

.. code-block:: console

    simtools-plot-tabular-data --plot_config config_file_name --output_file output_file_name

Plot tabular data for a specific type defined in the schema file:

.. code-block:: console

    simtools-plot-tabular-data \\
        --parameter atmospheric_profile \\
        --parameter_version 1.0.0 \\
        --site North \\
        --plot_type refractive_index_vs_altitude

Plot tabular data for all types defined in the schema file:

.. code-block:: console

    simtools-plot-tabular-data \\
        --parameter fadc_pulse_shape \\
        --parameter_version 1.0.0 \\
        --site North \\
        --telescope LSTN-01 \\
        --plot_type all
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: plot_tabular_data
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: plot_tabular_data_for_single_pe_from_ecsv_file.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: plot_tabular_data_for_single_pe_legacy_data.yml
```

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
