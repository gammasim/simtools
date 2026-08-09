# simtools-inspect-file

```{eval-rst}
.. automodule:: simtools.applications.inspect_file
   :members:
   :exclude-members: main
```

```{eval-rst}
For known simulation products, the application can append specialized
inspection sections on top of the generic file-structure report.

**Command line arguments**
input_file (str, required)
    Simulation-related file to inspect.
max_entries (int, optional)
    Maximum number of entries or preview lines to print. Use 0 for no limit.
show_entry (str, optional)
    Print the content of one HDF5 root dataset instead of the file structure report.
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: inspect_file
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: inspect_file_cfg.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: inspect_file_hdf5.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: inspect_file_simtel.yml
```
