# simtools-compare-productions

```{eval-rst}
.. automodule:: simtools.applications.compare_productions
   :members:
   :exclude-members: main
```

## Overview

This application compares trigger-histogram products from two or more simulation productions at
the event level. Each production is identified by a label and one or more comma-separated input
file patterns.

The current implementation supports the `events` comparison level. The `signals` and `compute`
levels are reserved for future comparison implementations. Use `array_layout_name` to restrict
the comparison to selected layouts.

## Input and output

| Role | Argument or file | Description |
| --- | --- | --- |
| Input | `production` | Repeated label and trigger-histogram pattern pairs. |
| Input | `array_layout_name` | Optional array-layout selection. |
| Output | `output_path` | Comparison plots grouped by array layout. |

Event-level comparisons include trigger multiplicity, trigger combinations, and telescope
participation fractions.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: compare_productions
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: compare_productions_run.yml
```
