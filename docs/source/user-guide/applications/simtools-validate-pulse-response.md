# simtools-validate-pulse-response

```{eval-rst}
.. automodule:: simtools.applications.validate_pulse_response
   :members:
   :exclude-members: main
```

Measure configured high- and low-gain pulse tables versus illumination in p.e.
The PDF contains peak ADC, full-window ADC sum, width, and waveform panels;
ECSV files contain individual measurements and summary values. A clipped-linear
control uses the same model gain and digitizer limit.

The calculation samples the input tables at the model FADC interval. Non-linear
amplitudes already encode the complete response. Interpolation is linear in the
input tables; it does not reproduce internal sim_telarray resampling or noise.
The conventional pulse is peak-normalized before transit-time-jitter smearing;
the non-linear tables already include jitter.

Widths use 95% of the peak by default. Set an absolute ADC threshold to measure
time-over-threshold instead. A missing threshold crossing produces a truncation
flag and an undefined width. ADC sums retain negative pulse tails.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: validate_pulse_response
   :no-heading:
```

## Workflow

Run the configurations under `tests/integration_tests/config/` with
`simtools-run-application --config_file <workflow.yml> --replace __OUTPUT__=<directory>`.
The direct-injection and full-simulation configurations run the existing flasher
application followed by the analyzer. Use more events and intensity levels for
statistical studies. Synthetic response tables test software behavior; comparison
with measured curves requires measured tables and matching camera settings.

```{eval-rst}
.. simtools-integration-example::
   :file: validate_pulse_response.yml
```
