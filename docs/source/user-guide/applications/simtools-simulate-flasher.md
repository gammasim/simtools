# simtools-simulate-flasher

```{eval-rst}
.. automodule:: simtools.applications.simulate_flasher
   :members:
   :exclude-members: main
```

```{eval-rst}
The flasher simulation allows for two different run modes:

1. Direct injection of light into the camera (bypassing the telescope optics).
2. Simulation of the full light path (using the light-emission package from sim_telarray).

The direct injection mode uses a simplified model for the flasher light source. Both run modes
provide events in sim_telarray format that can be processed by standard analysis steps or
visualized using e.g. the 'simtools-plot-simtel-events' application.

**Example Usage**

1. Simulate a single telescope:

    .. code-block:: console

        simtools-simulate-flasher --run_mode full_simulation \
        --light_source_type flat_fielding --model_version 7.0.0 \
        --site North --telescopes MSTN-04 --run_number 10

2. Simulate several telescopes:

    .. code-block:: console

        simtools-simulate-flasher --run_mode full_simulation \
        --light_source_type flat_fielding --model_version 7.0.0 \
        --site North --telescopes MSTN-04 MSTN-05 --run_number 10

3. Simulate all telescopes from an array layout:

    .. code-block:: console

        simtools-simulate-flasher --run_mode full_simulation \
        --light_source_type flat_fielding --model_version 7.0.0 \
        --site North --array_layout_name alpha --run_number 10

4. Simulate flashers for direct injection:

    .. code-block:: console

        simtools-simulate-flasher --run_mode direct_injection \
        --light_source MSFx-FlashCam --model_version 7.0.0 \
        --array_layout_name subsystem_msts --site South \
        --run_number 3

**Command Line Arguments**

run_mode (str, required)
    Run mode, either "direct_injection" or "full_simulation".
telescopes (str, optional)
    One or more telescope names (e.g. LSTN-01, MSTN-04, SSTS-04, ...).
    Use for single-telescope or multi-telescope simulations.
array_layout_name (str, optional)
    Name of the array layout. In full-simulation mode, all telescopes from this layout
    are simulated (one run per telescope).
site (str, required)
    Site name (North or South).
light_source (str, optional)
    Explicit calibration light source model, e.g. MSFx-FlashCam.
light_source_type (str, optional)
    Light source type, e.g. flat_fielding. Recommended for array-style simulations
    because the corresponding flasher model is read from the model-parameter database
    for each telescope.
number_of_events (int, optional):
    Number of events to simulate (default: 1). Can be a single value or a list.
flasher_photons (int, optional)
    Overwrite the model parameter flasher_photons. Can be a single value or
    a list for filter wheel sequences.
model_version (str, optional)
    Version of the simulation model.
run_number (int, optional)
    Run number to use (default: 1, required for direct injection mode).
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: simulate_flasher
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_flasher_direct_injection_lst_north.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_flasher_direct_injection_lst_north_filter_wheel.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_flasher_direct_injection_lst_south.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_flasher_direct_injection_mst_flashcam.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_flasher_direct_injection_mst_nectarcam.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_flasher_full_simulation_alpha_north.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_flasher_full_simulation_lst_filter_wheel.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_flasher_full_simulation_lst_filter_wheel_single_event_value.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_flasher_full_simulation_lst_south.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_flasher_full_simulation_mst_flashcam.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_flasher_full_simulation_mst_nectarcam.yml
```
