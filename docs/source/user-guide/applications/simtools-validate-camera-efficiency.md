# simtools-validate-camera-efficiency

```{eval-rst}
.. automodule:: simtools.applications.validate_camera_efficiency
   :members:
   :exclude-members: main
```

```{eval-rst}
Uses the sim_telarray tool "testeff" to calculate the camera efficiency.
The results of telescope throughput including optical and camera components for Cherenkov (left)
and NSB light (right) as a function of wavelength are plotted. See examples below.

.. _validate_camera_eff_plot:
.. image:: images/validate_camera_efficiency_North-MST-NectarCam-D_cherenkov.png
  :width: 49 %
.. image:: images/validate_camera_efficiency_North-MST-NectarCam-D_nsb.png
  :width: 49 %

**Command line arguments**

site (str, required)
    North or South.
telescope (str, required)
    Telescope model name (e.g. LSTN-01, SSTS-15)
model_version (str, optional)
    Simulation model version
zenith_angle (float, optional)
    Zenith angle in degrees (between 0 and 180).
azimuth_angle (float, optional)
    Telescope pointing direction in azimuth.
nsb_spectrum (str, optional)
    File with NSB spectrum to use for the efficiency simulation.

**Example**

MSTN-01 5.0.0

Runtime < 1 min.

.. code-block:: console

    simtools-validate-camera-efficiency --site North \\
        --azimuth_angle 0 --zenith_angle 20 \\
        --nsb_spectrum average_nsb_spectrum_CTAO-N_ze20_az0.txt \\
        --telescope MSTN-01 --model_version 5.0.0

The output is saved in simtools-output/validate_camera_efficiency.
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: validate_camera_efficiency
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: validate_camera_efficiency_lstn-02.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: validate_camera_efficiency_lstn-02_overwrite.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: validate_camera_efficiency_lsts-02.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: validate_camera_efficiency_mstx_flashcam_south.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: validate_camera_efficiency_mstx_nectarcam_north.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: validate_camera_efficiency_ssts.yml
```
