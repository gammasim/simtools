# simtools-run-science-tests

```{eval-rst}
.. automodule:: simtools.applications.run_science_tests
   :members:
   :exclude-members: main
```

Run named workflows from the shared science-test catalogue with an explicit release
bundle and external context. See [Science tests](../../developer-guide/testing_science.md)
for the file layout, production gates, and result interpretation.

```console
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --context_file /path/to/context.yml --dry_run
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: run_science_tests
   :no-heading:
```
