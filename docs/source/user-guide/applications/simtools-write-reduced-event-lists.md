# simtools-write-reduced-event-lists

```{eval-rst}
.. automodule:: write_reduced_event_lists
   :members:
   :exclude-members: main
```

## Overview

The application processes one or more sim_telarray output files (``*.simtel.zst``)
and writes reduced event lists in HDF5 format. Input files can be given directly
on the command line or read from a text file and processed in batches.

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
