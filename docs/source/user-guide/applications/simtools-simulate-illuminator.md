# simtools-simulate-illuminator

```{eval-rst}
.. automodule:: simtools.applications.simulate_illuminator
   :members:
   :exclude-members: main
```

```{eval-rst}
Illuminators are calibration light sources not attached to a particular telescope.
Three modes of operation are supported:

1. Single pair: simulate one illuminator-telescope pair with positions from the model database.
2. Single pair with configurable position: override the illuminator position and pointing.
3. Multi-pair: simulate all valid illuminator-telescope pairs from the visibility table
   in parallel, using a configurable number of CPU cores.

**Example Usage**

1. Simulate illuminator with positions as defined in the simulation models database:

    .. code-block:: console

        simtools-simulate-illuminator --light_source ILLN-01 \
        --telescope MSTN-04 --site North \
        --model_version 7.0.0

2. Simulate at a configurable position (1km above array center) and pointing downwards:

    .. code-block:: console

        simtools-simulate-illuminator --light_source ILLN-01 \
        --light_source_position 0. 0. 1000. \
        --light_source_pointing 0. 0. -1. \
        --telescope MSTN-15 --site North \
        --model_version 7.0.0

3. Simulate all valid pairs from the visibility table in parallel:

    .. code-block:: console

        simtools-simulate-illuminator --site North \
        --model_version 7.0.0 --simulate_all

4. Simulate all pairs for a specific illuminator only:

    .. code-block:: console

        simtools-simulate-illuminator --site North \
        --model_version 7.0.0 --simulate_all \
        --light_source ILLN-01

5. Simulate all pairs with explicit worker count:

    .. code-block:: console

        simtools-simulate-illuminator --site North \
        --model_version 7.0.0 --simulate_all \
        --max_workers 8

6. Simulate with a specific wavelength (e.g., 355 nm):

    .. code-block:: console

        simtools-simulate-illuminator --light_source ILLN-01 \
        --telescope MSTN-04 --site North \
        --model_version 7.0.0 --wavelength 355 nm

7. Simulate with multiple wavelengths:

    .. code-block:: console

        simtools-simulate-illuminator --light_source ILLN-01 \
        --telescope MSTN-04 --site North \
        --model_version 7.0.0 --wavelength 355 nm 473 nm

8. Simulate all pairs for all wavelengths in model (no wavelength specified):

    .. code-block:: console

        simtools-simulate-illuminator --site North \
        --model_version 7.0.0 --simulate_all

9. Using a config file with specific wavelengths:

    Create a config file (e.g., illuminator_config.yml):

    .. code-block:: yaml

        site: North
        model_version: 7.0.0
        light_source: ILLN-01
        telescope: MSTN-04
        wavelength: [355, 473]

    Then run:

    .. code-block:: console

        simtools-simulate-illuminator --config illuminator_config.yml

**Command Line Arguments**
light_source (str, optional)
    Illuminator in array, e.g., ILLN-01. Required for single-pair mode.
    In multi-pair mode, used as a filter (simulate only pairs with this illuminator).
number_of_events (int, optional)
    Number of events to simulate.
flasher_photons (int, optional)
    Overwrite the model parameter flasher_photons.
wavelength (float or list of float, optional)
    Wavelength(s) in nanometers (unitless values are interpreted as nm).
    Must be one of the wavelengths supported by the illuminator model.
    Multiple wavelengths can be specified (space-separated on command line,
    or as a list in config file: wavelength: [355, 473]).
    If not specified, all model wavelengths will be simulated
    Each wavelength will be validated and simulated as a separate job.
telescope (str, optional)
    Telescope model name (e.g. LSTN-01, MSTN-04, ...). Required for single-pair mode.
    In multi-pair mode, used as a filter (simulate only pairs with this telescope).
site (str, required)
    Site name (North or South).
model_version (str, optional)
    Version of the simulation model.
simulate_all (flag, optional)
    Simulate all valid illuminator-telescope pairs from the visibility table.
max_workers (int, optional)
    Maximum number of parallel workers for multi-pair mode. Default: 60% of CPU cores.
    Set to 0 to use all available cores.
light_source_position (float, float, float, optional)
    Light source position (x,y,z) relative to the array center (ground coordinates) in
    m. If not set, the position from the simulation model is used.
light_source_pointing (float, float, float, optional)
    Light source pointing direction. If not set, the pointing from the simulation model is used.
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: simulate_illuminator
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_illuminator_configurable_position.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_illuminator_layout.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_illuminator_multi.yml
```
