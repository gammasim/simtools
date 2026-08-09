# simtools-simulate-pedestals

```{eval-rst}
.. automodule:: simtools.applications.simulate_pedestals
   :members:
   :exclude-members: main
```

```{eval-rst}
Use sim_telarray to simulate pedestal events for an array of telescopes.
The following types are supported:

* Pedestal events (includes night-sky background and camera noise)
* Dark pedestal events (closed camera lid, camera noise only)
* NSB-only pedestal events (open camera lid, night-sky background only, no camera noise)

**Example**

Simulate pedestal events for Alpha North. The assumed level night-sky background is 2.0 times the
nominal value. A list of stars can be provided to simulate additional contributions.

.. code-block:: console

    simtools-simulate-pedestals --run_mode=pedestals \\
        --run_number 10 --number_of_events 1000 \\
        --array_layout_name alpha --site North \\
        --model_version 6.0.0 \\
        --zenith_angle 20 --azimuth_angle 0 \\
        --nsb_scaling_factor 2.0


**Command Line Arguments**
run_mode (str, required)
    Run mode, e.g. "pedestals"
run_number (int, required)
    Run number for the simulation.
number_of_events (int, required)
    Number of calibration events to simulate.
array_layout_name (str, required)
    Array layout name, e.g. "alpha".
site (str, required)
    Site name.
model_version (str, optional)
    Version of the simulation model.
nsb_scaling_factor (float, optional)
    Scaling factor for the night-sky background rate. Default is 1.0, which
    corresponds to the nominal (dark sky) NSB rate.
stars (str, optional)
    Path to a file containing a list of stars (azimuth, zenith, weighting factor
    separated by whitespace). If provided, the stars will be used to simulate
    additional contributions to the night-sky background. For details on the
    parameters, see the sim_telarray manual.
zenith_angle (float, optional)
    Zenith angle in degrees.
azimuth_angle (float, optional)
    Azimuth angle in degrees.
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: simulate_pedestals
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_pedestals_20_deg_north.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_pedestals_dark_20_deg_north.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_pedestals_nsb_only_20_deg_north.yml
```
