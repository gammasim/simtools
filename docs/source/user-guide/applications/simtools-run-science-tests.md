# simtools-run-science-tests

```{eval-rst}
.. automodule:: simtools.applications.run_science_tests
   :members:
   :exclude-members: main
```

Run selected science tests using a release configuration and a separate file containing production
paths. See [Science tests](../../developer-guide/testing_science.md) for configuration files,
production requirements, and result interpretation.

When the campaign catalogue defines a camera-efficiency comparison, select the baseline and
candidate calculations together with the comparison. The comparison consumes their exported
artifacts and does not rerun either calculation:

```console
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --test camera_efficiency.baseline --test camera_efficiency.candidate \
  --test compare.camera_efficiency --dry_run
```

The comparison report includes a JSON result and scalar plots. See
[Compare science results](simtools-compare-science-results.md) for the result fields and
advisory tolerance behavior.

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
