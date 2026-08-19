# simtools-plot-tabular-data

```{eval-rst}
.. automodule:: simtools.applications.plot_tabular_data
   :members:
   :exclude-members: main
```

```{eval-rst}
Uses a configuration file to define the data to be plotted and all
plotting details.

**Command line arguments**

config_file (str, required)
    Configuration file name for plotting.
output_file (str, required)
    Output file name (without suffix).

**Example**

Plot tabular data using a configuration file.

.. code-block:: console

    simtools-plot-tabular-data --plot_config config_file_name --output_file output_file_name
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
