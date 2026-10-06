# simtools-write-reduced-event-lists

This application supports the `local` (default) and `htcondor` execution backends. See
[Execution backends](../execution_backends.md) for HTCondor setup and configuration.

```{eval-rst}
.. automodule:: simtools.applications.write_reduced_event_lists
   :members:
   :exclude-members: main
```

## Overview

Converts sim_telarray event files (``*.simtel.zst``) into a
compact, analysis-oriented HDF5 product. It removes waveform- and pixel-level
data while retaining the event-level quantities needed for trigger-rate,
effective-area, and Monte Carlo statistics studies. Input files can be supplied
directly, through one text file, or through a glob pattern matching several text
files; several input files can be combined into one output file.

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
| Input | `input_files` | sim_telarray | One or more `*.simtel.zst` files. |
| Input | `input_file_list` | Text file | One sim_telarray output file per line. |
| Input | `input_file_list_pattern` | Glob pattern | Text files containing one sim_telarray output file per line. |
| Output | `output_path` | Directory | Reduced-event HDF5 files with embedded metadata. |

Provide exactly one of `input_files`, `input_file_list`, or `input_file_list_pattern`. A pattern
processes each matching list independently and submits all resulting output batches together. Use
`max_workers` to control parallel output-file processing.

If an HDF5 write fails, the retained incomplete file includes the application activity ID and
the per-write staging ID in its name. The activity ID matches the UUID in the generated
`write_reduced_event_lists_<activity_id>.log` file.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: write_reduced_event_lists
   :no-heading:
```

## Example

```bash
python src/simtools/applications/write_reduced_event_lists.py \
    --input_file_list_pattern "/lustre/fs25/group/cta/prod6/north/reduced_event_list/*.txt" \
    --output_path /lustre/fs25/group/cta/prod6/north/reduced_event_list \
    --files_per_reduced_event_file 10 \
    --backend htcondor \
    --backend_config htcondor.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: write_reduced_event_lists.yml
```
