# simtools-plot-simulated-event-distributions

```{eval-rst}
.. automodule:: simtools.applications.plot_simulated_event_distributions
   :members:
   :exclude-members: main
```

```{eval-rst}
Reads reduced event data files and generate histogram plots e.g. for energy or
core distance distributions.

**Command line arguments**
trigger_histogram_file (str, required)
    Precomputed trigger-histogram HDF5 file from ``simtools-write-trigger-histograms``.
array_layout_name (str, optional)
    Optional array layout name to select from a precomputed trigger-histogram HDF5 file.
output_path (str, required)
    Output directory for the generated plots.

**Examples**
Generate plots from a precomputed trigger-histogram file:

.. code-block:: console

    simtools-plot-simulated-event-distributions \
        --trigger_histogram_file trigger_histograms.hdf5 \
        --array_layout_name alpha \
        --output_path simtools_output
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: plot_simulated_event_distributions
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: plot_simulated_event_distributions.yml
```
