# simtools-compare-productions

```{eval-rst}
.. automodule:: simtools.applications.compare_productions
   :members:
   :exclude-members: main
```

## Overview

This application compares trigger-histogram HDF5 products from two or more simulation
productions at the event level. Each production is identified by a label and one or more
comma-separated input file patterns. Multiple files belonging to one label are aggregated.

The first production is the baseline. Every following production is compared with that baseline,
so at least two production descriptors are required and production order matters. Production
labels must be unique.

The application currently performs event-level comparisons only. Trigger-histogram files should
normally be produced with
[simtools-write-trigger-histograms](simtools-write-trigger-histograms).

Use `array_layout_name` to restrict the comparison to selected layouts. Without this option, all
array-layout references found in the input files are aggregated into one comparison. With multiple
selected layouts, each layout is compared independently and written to its own directory below
`output_path/<layout-name>/`.

## Input and output

| Role | Argument or file | Format | Description |
| --- | --- | --- | --- |
| Input | `production` | HDF5 | Repeated label and comma-separated file-pattern pairs. |
| Input | `array_layout_name` | Array-layout name | Optional selection of references to aggregate. |
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
    :file: compare_productions_run.yml
```
