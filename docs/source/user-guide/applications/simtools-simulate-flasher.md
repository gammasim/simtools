# simtools-simulate-flasher

```{eval-rst}
.. automodule:: simtools.applications.simulate_flasher
   :members:
   :exclude-members: main
```

```{eval-rst}
The flasher simulation allows for two different run modes:

1. Direct injection of light into the camera (bypassing the telescope optics).
2. Simulation of light emission from the source using the light-emission package from sim_telarray.

Full-simulation flat-fielding uses the source geometry and angular distribution to illuminate
the camera, but sets ``Bypass_Optics=1`` in sim_telarray, bypassing telescope optics in this case.
Both modes produce sim_telarray events for standard analysis or
visualization with 'simtools-plot-simtel-events'.

**Photon counts and matching run modes**

The ``flasher_photons`` setting has a different meaning in each run mode:

.. list-table:: Photon count meanings by run mode
   :header-rows: 1

   * - Run mode
     - Meaning of ``flasher_photons``
   * - ``direct_injection``
     - Overrides the model parameter ``flasher_photons_at_pixel``, setting the injected photon
       count per pixel.
   * - ``full_simulation``
     - Sets the number of photons emitted by the flasher source and passed to sim_telarray.

These values are not interchangeable: source geometry and angular distribution determine the
full-simulation illumination, while direct injection uses one uniform value per pixel. Equal
numeric values do not produce a matched simulation, and dividing by the number of camera pixels
does not provide a general conversion.

For a matched comparison, determine the expected camera illumination for the chosen source,
wavelength, pulse, and telescope, then derive the direct per-pixel setting. Validate it with signal
distributions and a separate per-pixel mean and variance analysis; the comparison below summarizes
signals across pixels and events.

**Comparing signal outputs**

Compare the sim_telarray files with ``simtools-compare-productions`` in signal mode. Quote file
patterns so the application can expand them:

.. code-block:: console

    simtools-compare-productions \
        --comparison_level signal \
        --production direct_injection "/path/to/direct/*.simtel.zst" \
        --production full_simulation "/path/to/full/*.simtel.zst" \
        --telescope_name LSTN-01 \
        --output_path flasher-comparison

The ``signals.png`` plot compares integrated-signal distributions across pixels and events.
Its legend identifies productions only; it does not display means or match pixels between files.
For mean-based scaling, use
``simtools.sim_events.production_comparison.collect_signal_metrics`` and calculate
``metrics.signals.mean()`` for each returned production metric, grouped by telescope. See
:doc:`simtools-compare-productions` for comparison options and outputs.

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
    because the corresponding flasher model is read from the simulation-model repository
    for each telescope.
number_of_events (int, optional):
    Number of events to simulate (default: 1). Can be a single value or a list.
flasher_photons (int, optional)
    Photon intensity; its meaning depends on the run mode as described above. Can be a single
    value or a list for filter wheel sequences.
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
