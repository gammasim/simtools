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

Use `--disable_nsb_autoscaling` to disable the zenith-dependent NSB correction. A manual `--nsb_scaling_factor` still applies.
