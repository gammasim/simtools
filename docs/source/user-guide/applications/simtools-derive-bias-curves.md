# simtools-derive-bias-curves

```{eval-rst}
.. automodule:: simtools.applications.derive_bias_curves
   :members:
   :exclude-members: main
```

```{eval-rst}
**Description**

This application combines NSB (Night Sky Background) and proton trigger rates
to generate bias curves showing how trigger rates vary with threshold.

The tool:
1. Extracts NSB trigger rates from reduced event-data HDF5 files
2. Calculates proton trigger rates from proton reduced event-data HDF5 files
3. Plots both curves on the same figure for comparison
4. Outputs ecsv files for runwise nsb simulation,
runwise proton simulation, nsb rate and proton rate vs threshold

The input directory should contain both:
- NSB reduced event-data HDF5 files
- Proton simulation reduced event-data HDF5 files

The input files can be generated using simtools-generate-bias-curve-submissions.

**Command line arguments**

data_dir (str, required)
    Directory containing NSB/proton reduced event-data HDF5 files (e.g. gamma* and proton*).
figure_file (str, optional)
    Output plot file path or output directory. Default: bias_curve.png
nsb_table_file (str, optional)
    Output ECSV table file for NSB trigger rates. If not specified, no table is written.
proton_table_file (str, optional)
    Output ECSV table file for proton rates. If not specified, no table is written.
site (str, required)
    Site name (North/South) for telescope configuration.
model_version (str, required)
    Model version for telescope configuration.
telescope (str, required)
    Telescope name for configuration.
title (str, optional)
    Plot title. Default: "Trigger Rate Bias Curves"
ymin (float, optional)
    Minimum y-axis value for plot. Default: 1e2
ymax (float, optional)
    Maximum y-axis value for plot. Default: 5e5

**Example**

.. code-block:: console

    simtools-derive-bias-curves \\
        --data_dir /path/to/data \\
        --site North \\
        --model_version 7.0.0 \\
        --telescope LSTN-01 \\
        --figure_file bias_curves.png
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: derive_bias_curves
   :no-heading:
```
