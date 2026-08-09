# simtools-derive-trigger-rates

```{eval-rst}
.. automodule:: simtools.applications.derive_trigger_rates
   :members:
   :exclude-members: main
```

```{eval-rst}
Uses simulated background events (e.g. from proton primaries) to calculate the trigger rates.
Input is reduced event data generated from simulations for the given configuration.


**Command line arguments**
event_data_file (str, required)
    Event data file containing reduced event data.
array_layout_name (list, optional)
    Name of the array layout to use for the simulation.
telescope_config_file (str, optional)
    Path to a file containing telescope configurations.
plot_histograms (bool, optional)
    Plot histograms of the event data.
model_version (str, optional)
    Version of the simulation model to use.
site (str, optional)
    Name of the site where the simulation is being run.
cr_spectrum (str, optional)
    Path to a YAML file defining a user-provided cosmic-ray spectrum.
    Supported spectrum types: PowerLaw, LogParabola, PowerLawWithExponentialGaussian.
    If not given, the spectrum is selected from the CTAO spectrum library.


**Example**

Derive trigger rates for the South Alpha layout:

.. code-block:: console

    simtools-derive-trigger-rates \\
        --site South \\
        --model_version 6.0.0 \\
        --event_data_file /path/to/event_data_file.h5 \\
        --array_layout_name alpha\\
        --plot_histograms

Derive trigger rates with a user-defined spectrum:

.. code-block:: console

    simtools-derive-trigger-rates \\
        --site South \\
        --model_version 6.0.0 \\
        --event_data_file /path/to/event_data_file.h5 \\
        --array_layout_name alpha \\
        --cr_spectrum /path/to/spectrum.yml
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: derive_trigger_rates
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: derive_trigger_rates_db_arrays_short.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: derive_trigger_rates_user_spectrum.yml
```
