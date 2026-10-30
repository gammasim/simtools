# simtools-plot-simtel-events

```{eval-rst}
.. automodule:: simtools.applications.plot_simtel_events
   :members:
   :exclude-members: main
```

Use `--pedestal_source monitoring` to use the per-pixel high-gain pedestal stored
in sim_telarray camera monitoring data for pedestal plots and signal integration.
The default, `waveform`, estimates the pedestal separately for each event from the
last 10 waveform samples; a pulse tail in that window can bias the estimate.
The monitoring option requires valid monitoring data for every selected event.

```{eval-rst}
Produces diagnostic figures from sim_telarray (.simtel.zst) files.
Meant to run after simulations (e.g., simtools-simulate-flasher,
simtools-simulate-illuminator).

**What it does**

- Loads the provided sim_telarray file
- Generates selected plots (signals, pedestals, time traces, waveforms, peak timing, etc.)
- Saves all figures to a single multi-page PDF
- Optionally also saves individual PNG files per figure

**Command line arguments**

simtel_file (str, required)
    A sim_telarray file to visualize (.simtel.zst).
telescope (str, required)
    Telescope name to process (e.g., LSTN-04, MSTN-01).
plots (list, optional)
    Which plots to generate. Choices: pedestals, signals, peak_timing, time_traces,
    waveforms, step_traces, all. Default: all.
number_of_pixels (int, optional)
    For time_traces: number of brightest pixel traces to plot. Default: 3.
pixel_step (int, optional)
    For step_traces and waveforms: step between pixel indices. Default: 100.
max_pixels (int, optional)
    For step_traces: maximum number of pixels to plot. Default: None (no limit).
vmax (float, optional)
    For waveforms: upper limit of color scale. Default: None (auto-scale).
sum_threshold (float, optional)
    For peak_timing: minimum pixel sum to consider. Default: 10.0.
timing_bins (int, optional)
    For peak_timing: number of histogram bins. Default: None (unit-width bins).
event_id (int or list, optional)
    Specific event ID(s) to plot. Default: None (first event).
max_events (int, optional)
    Maximum number of events to process. Default: 1.
pedestal_source (str, optional)
    Pedestal source for pedestal plots and signal integration. Choices: waveform,
    monitoring. Default: waveform.
output_file (str, optional)
    Base name for output files. PDF will be named ``<base>_<inputstem>.pdf``.
    If omitted, uses input file stem.
save_pngs (flag, optional)
    Also save individual PNG files per plot.
figure_dpi (int, optional)
    Resolution for PNG outputs (alias: ``dpi``). Default: 300.
output_path (str, optional)
    Directory for output files.

**Examples**

1) Plot signals and time traces for a telescope:

   simtools-plot-simtel-events \\
     --simtel_file run000010_North_7.0.0_simulate_flasher.simtel.zst \\
     --telescope LSTN-04 \\
     --plots signals time_traces \\
     --output_file flasher_inspect

2) Generate all plots with PNG outputs:

   simtools-plot-simtel-events \\
     --simtel_file run000010.simtel.zst \\
     --telescope MSTN-01 \\
     --plots all \\
     --save_pngs --figure_dpi 200

3) Plot specific events:

   simtools-plot-simtel-events \\
     --simtel_file run000010.simtel.zst \\
     --telescope LSTN-04 \\
     --event_id 5 10 15 \\
     --plots signals pedestals
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: plot_simtel_events
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: plot_simtel_events.yml
```
