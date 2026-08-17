# simtools-derive-incident-angle

```{eval-rst}
.. automodule:: simtools.applications.derive_incident_angle
   :members:
   :exclude-members: main
```

```{eval-rst}
Creates photon files with additional columns for incident angles calculation.
Outputs files and histograms of the incidence angles at
the focal plane, primary mirror, and if available, secondary mirror.
Optional debug plots can be also generated.
Note that this application does not include a full raytracing of telescope structures,
and their non symmetric shadowing at off-axis angles.

**Example usage**

.. code-block:: console

    simtools-derive-incident-angle \
        --off_axis_angles 0 1 2 3 4 \
        --source_distance 10 \
        --number_of_photons 1000000 \
        --model_version 6.0.0 \
        --telescope MSTN-04 \
        --site North

**Command line arguments**

- off_axis_angles (float, optional)
    One or more off-axis angles in degrees (space-separated). Default: [0.0].
- source_distance (float, optional)
    Source distance in kilometers. Default: 10.0.
- number_of_photons (int, optional)
    Number of photons of the light source to trace per run. Default: 10000.
- perfect_mirror (flag, optional)
    Assume perfect mirror shape/alignment/reflection.
- debug_plots (flag, optional)
    Generate additional debug plots (radius histograms, XY heatmaps, radius vs angle).
- calculate_primary_secondary_angles / no-calculate_primary_secondary_angles
    Include or skip angles on primary/secondary mirrors. Default: include.

The application writes:

- imaging list (photons) file
- stars list file
- a histogram of incident angles (PNG)
- a results table in ECSV format

Example of a focal-plane incident angle plot for a SST:

.. _plot_derive_incident_angle_plot:
.. image:: images/incident_angles_multi_derive_incident_angle_SSTS-04.png
    :width: 49 %

Example of a primary mirror incident angle plot for a SST:

.. _plot_derive_incident_angle_plot_primary:
.. image:: images/incident_angles_primary_multi_derive_incident_angle_SSTS-04.png
    :width: 49 %

Note also the relation between radius and primary mirror incident angles, and how this relates to
the peak seen in the primary mirror incident angle distribution:

.. _plot_derive_incident_angle_plot_angle_vs_radius:
.. image:: images/primary_angle_vs_radius.png
    :width: 49 %

Example of a secondary mirror incident angle plot for a SST:

.. _plot_derive_incident_angle_plot_secondary:
.. image:: images/incident_angles_secondary_multi_derive_incident_angle_SSTS-04.png
    :width: 49 %
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: derive_incident_angle
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: derive_incident_angle_run_dual_mirror.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: derive_incident_angle_run_dual_mirror_debug.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: derive_incident_angle_run_dual_mirror_debug_offsets.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: derive_incident_angle_run_single_mirror.yml
```
