# simtools-write-reduced-event-lists

```{eval-rst}
.. automodule:: write_reduced_event_lists
   :members:
   :exclude-members: main
```

## Overview

Converts sim_telarray event files (``*.simtel.zst``) into a
compact, analysis-oriented HDF5 product. It removes waveform- and pixel-level
data while retaining the event-level quantities needed for trigger-rate,
effective-area, and Monte Carlo statistics studies. Input files can be supplied
directly or through a text file; several input files can be combined into one
output file.

Each output file contains the following root-level datasets:

- ``SHOWERS``: simulated energy, core position, shower direction, event IDs, and
  area weights for shower records.
- ``TRIGGERS``: array pointing and the telescope lists for triggered records.
- ``FILE_INFO``: one row per input file with file IDs and simulation settings,
  including particle, energy, viewcone, core-scatter, pointing, and NSB values.
- ``METADATA``: standard simtools product metadata.
- ``SIMULATION_METADATA``: versioned provenance for the input files and simulation
  software.

The ``file_id`` column joins ``SHOWERS`` and ``TRIGGERS`` to ``FILE_INFO``;
``TRIGGERS`` can contain fewer rows than ``SHOWERS`` because it records only
triggered events. Tables are stored as HDF5 compound datasets, with physical
columns carrying their Astropy units.

## Input and output

| Role | Argument | Format | Description |
| --- | --- | --- | --- |
| Input | `input_files` | sim_telarray | One or more `*.simtel.zst` files or glob patterns. |
| Output | `output_path` | Directory | Reduced-event HDF5 files with embedded metadata. |

Provide either `input_files` or `input_file_list`, but not both. Use `max_workers` to control
parallel output-file processing.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: write_reduced_event_lists
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: write_reduced_event_lists.yml
```
