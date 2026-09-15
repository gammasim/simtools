# simtools-compare-productions

```{eval-rst}
.. automodule:: simtools.applications.compare_productions
   :members:
   :exclude-members: main
```

## Overview

This application compares simulation products from two or more productions at the event or signal
level, or summarizes CORSIKA and sim_telarray resource requirements for a production. For event
and signal comparisons, each production is identified by a label and one or more comma-separated
input file patterns. Multiple files belonging to one label are aggregated.

The first production is the baseline. Every following production is compared with that baseline,
so at least two production descriptors are required and production order matters. Production
labels must be unique.

The application supports event-level and signal-level comparisons. Trigger-histogram files should
normally be produced with
[simtools-write-trigger-histograms](simtools-write-trigger-histograms).

For production resource requirements, use `--comparison_level compute` with a production root in
`--baseline_path`. This mode discovers selected job manifests and their CORSIKA and sim_telarray
resource records. It writes `resource_requirements.ecsv` (one normalized process row per job), a
grouped `resource_requirements.md` report, and time, memory, and storage plots. Time, CPU, and
output sizes are normalized by `showers_per_run`; peak resident memory remains a per-process
maximum. Sim_telarray storage includes simtel event files, reduced event data, and histogram files.
Piped CORSIKA jobs have no retained CORSIKA output size and report it as missing. An optional
`--candidate_path` overlays a second production. Repeated `--select` expressions filter both
production manifests.

```console
simtools-compare-productions \
    --comparison_level compute \
    --baseline_path /data/production \
    --select configuration.primary=gamma \
    --output_path resource-requirements
```

For signal-level comparisons, use `--comparison_level signal` with sim_telarray files. By default,
all telescopes shared by the input files are processed. Use `--array_layout_name` with one or more
telescope names to restrict the comparison, for example:

```console
simtools-compare-productions \
    --comparison_level signal \
    --array_layout_name LSTN-01 MSTN-01 \
    --production flat /data/flat/run.simtel.zst \
    --production curved /data/curved/run.simtel.zst \
    --output_path signal-comparisons
```

`--telescope_name` is also accepted as an alias for signal-level telescope selection. Use only
one of these options at a time.

For event-level comparisons, use `array_layout_name` to restrict the comparison to selected layouts.
Without this option, all
array-layout references found in the input files are aggregated into one comparison. With multiple
selected layouts, each layout is compared independently and written to its own directory below
`output_path/<layout-name>/`.

For production metadata manifests, pass the baseline and candidate metadata directories instead of
legacy file descriptors. Repeated `select` expressions choose configurations, while `compare_by`
lists configuration fields that are allowed to differ. Each matched configuration is written below
`output_path/comparison-<configuration-hash>/`.

```console
simtools-compare-productions \
    --baseline_path /data/baseline/trigger_histograms \
    --candidate_path /data/candidate/trigger_histograms \
    --select configuration.primary=gamma \
    --compare_by configuration.atmosphere \
    --array_layout_name CTAO-North-Alpha \
    --output_path comparisons
```

## Input and output

| Role | Argument or file | Format | Description |
| --- | --- | --- | --- |
| Input | `production` | HDF5 | Repeated label and comma-separated file-pattern pairs. |
| Input | `array_layout_name` | Array-layout or telescope name(s) | Optional selection for event-level layouts or signal-level telescopes. |
| Input | `telescope_name` | Telescope name(s) | Optional alias for signal-level telescope selection. |
| Output | `output_path` | Directory | Directory for figures and the statistics report. |
| Output | `comparison_statistics.json` | JSON | Machine-readable comparison statistics. |
| Output | `comparison_statistics.meta.yml` | YAML | Metadata sidecar for the statistics report. |

Event-level comparisons include trigger multiplicity, trigger combinations, single-telescope and
mixed-type trigger distributions, telescope participation, and simulated/triggered distributions
of primary energy, core distance, and angular distance. Cumulative distributions and
per-telescope-type plots are also written when the corresponding input data are available.
Individual figures may be skipped when their input data are absent.

Figures are written as PNG files by default. Use `--figure_format pdf` to write PDF files, or
`--figure_format png pdf` to write both formats. Other formats supported by Matplotlib can also
be selected.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: compare_productions
   :no-heading:
```

## Integration example

```{eval-rst}
.. simtools-integration-example::
    :file: compare_productions_events.yml
```
