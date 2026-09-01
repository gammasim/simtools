# simtools-plot-corsika-limits

```{eval-rst}
.. automodule:: simtools.applications.plot_corsika_limits
   :members:
   :exclude-members: main
```

```{eval-rst}
This application reads a CORSIKA limits table and plots the limits as a
function of zenith angle. Azimuth directions share each plot: NSB levels use
different colors, while azimuths use solid or dashed lines with filled or open markers.


**Command line arguments**

corsika_limits_file (str, required)
    Path to a CORSIKA limits table in ECSV format.

**Example**

.. code-block:: console

   simtools-plot-corsika-limits \
       --corsika_limits_file simtools-output/merged_corsika_limits.ecsv \
       --output_path simtools-output
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: plot_corsika_limits
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: production_plot_corsika_limits.yml
```
