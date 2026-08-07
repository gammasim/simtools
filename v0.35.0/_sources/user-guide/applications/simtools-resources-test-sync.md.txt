# simtools-resources-test-sync

```{eval-rst}
.. automodule:: resources_test_sync
   :members:
   :exclude-members: main
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: resources_test_sync
   :no-heading:
```

## Examples

Generate a dry-run report of new, changed, and obsolete files in the
`tests/resources` directory (no files are copied or deleted):

```console
simtools-resources-test-sync \
    --test_directory ../simtools-tests \
    --simtools_version v0.34.0
```

To sync the test resources, add the `--sync` option. To list obsolete files
that should be removed manually, add the `--delete_missing` option.
