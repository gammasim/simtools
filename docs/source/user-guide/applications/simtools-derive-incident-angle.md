# simtools-derive-incident-angle

```{eval-rst}
.. automodule:: simtools.applications.derive_incident_angle
   :members:
   :exclude-members: main
```

Creates photon files with additional columns for incident angles calculation.
Outputs files and histograms of the incidence angles at the focal plane, primary mirror,
and if available, secondary mirror. Optional debug plots can be also generated.

Note that this application does not include a full raytracing of telescope structures,
and their non-symmetric shadowing at off-axis angles.

Example of a focal-plane incident angle plot for a SST:

```{eval-rst}
.. _plot_derive_incident_angle_plot:
.. image:: images/incident_angles_multi_derive_incident_angle_SSTS-04.png
    :width: 49 %
```

Example of a primary mirror incident angle plot for a SST:

```{eval-rst}
.. _plot_derive_incident_angle_plot_primary:
.. image:: images/incident_angles_primary_multi_derive_incident_angle_SSTS-04.png
    :width: 49 %
```

Note also the relation between radius and primary mirror incident angles, and how this relates to
the peak seen in the primary mirror incident angle distribution:

.. _plot_derive_incident_angle_plot_angle_vs_radius:
.. image:: images/primary_angle_vs_radius.png
    :width: 49 %

Example of a secondary mirror incident angle plot for a SST:

```{eval-rst}
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
    :file: derive_incident_angle_run_single_mirror.yml
```
