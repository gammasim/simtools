# simtools-validate-telescope-throughput

```{eval-rst}
.. automodule:: simtools.applications.validate_telescope_throughput
   :members:
   :exclude-members: main
```

```{eval-rst}
Calculates overall telescope throughput from camera and optical efficiencies, including telescope
shadowing, and derives the reference night-sky-background pixel rate. Plots show throughput for
Cherenkov (left) and night-sky-background light (right) as a function of wavelength.

.. _validate_telescope_throughput_plot:
.. image:: images/validate_telescope_throughput_North-MST-NectarCam-D_cherenkov.png
  :width: 49 %
.. image:: images/validate_telescope_throughput_North-MST-NectarCam-D_nsb.png
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

    simtools-validate-telescope-throughput --site North \\
        --azimuth_angle 0 --zenith_angle 20 \\
        --nsb_spectrum average_nsb_spectrum_CTAO-N_ze20_az0.txt \\
        --telescope MSTN-01 --model_version 5.0.0

The output is saved in simtools-output/validate_telescope_throughput.
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: validate_telescope_throughput
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: validate_telescope_throughput_lstn-02.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: validate_telescope_throughput_lstn-02_overwrite.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: validate_telescope_throughput_lsts-02.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: validate_telescope_throughput_mstx_flashcam_south.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: validate_telescope_throughput_mstx_nectarcam_north.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: validate_telescope_throughput_ssts.yml
```
