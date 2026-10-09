# simtools-run-science-tests

```{eval-rst}
.. automodule:: simtools.applications.run_science_tests
   :members:
   :exclude-members: main
```

Run selected science tests using a release configuration and a separate file containing production
paths. See [Science tests](../../developer-guide/testing_science.md) for configuration files,
production requirements, and result interpretation.

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
