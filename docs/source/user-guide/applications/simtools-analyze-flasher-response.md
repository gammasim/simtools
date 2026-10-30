# simtools-analyze-flasher-response

```{eval-rst}
.. automodule:: simtools.applications.analyze_flasher_response
   :members:
   :exclude-members: main
```

Analyze raw HG and LG calibration waveforms from a flasher intensity sequence.
The application writes per-event and summary ECSV tables and a PDF containing
peak, ADC sum, width, and mean-waveform plots. Error bars show event-to-event
standard deviations; the peak curve is the mean of individual-event peaks.

Supply one file per intensity, or one file containing all steps in declared
order. `events_per_level` must match the actual counts; one value broadcasts to
all steps. Analyze only scan files containing the declared calibration events.
The illumination axis is the requested photon count, not detected p.e.

Monitoring pedestals are converted to per-sample values separately for both
gains. If monitoring is unavailable, provide an explicit per-sample pedestal.
There is no automatic baseline estimate from possibly saturated pulse tails.
A pulse lacking width crossings inside the readout window is flagged as truncated;
its ADC sum covers only the recorded window.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: analyze_flasher_response
   :no-heading:
```

## Workflow

Run the configurations under `tests/integration_tests/config/` with
`simtools-run-application --config_file <workflow.yml> --replace __OUTPUT__=<directory>`.
The direct-injection and full-simulation configurations run the existing flasher
application followed by the analyzer. Use more events and intensity levels for
statistical studies. Synthetic response tables test software behavior; comparison
with measured curves requires measured tables and matching camera settings.

The saturation scan workflows record a longer readout window to include broadened
pulses. For whole-camera bright flashes, set `MAX_PHOTOELECTRONS=25000000` before
running the workflow; the integration test sets this capacity automatically.
The full-simulation smoke test uses photon bunches of 100 for speed. Use bunches
of one and more events when studying photon statistics or comparing measurements.
The direct-injection workflow also writes a matched linear/clipped control under
`linear/`, with the non-linear pulse tables disabled.

```{eval-rst}
.. simtools-integration-example::
   :file: validate_saturation_direct_injection_lst.yml
```
