# Science tests

Science tests provide extensive physics validation for new releases of `simtools`,
`CORSIKA`, `sim_telarray` and any other `simtools` dependency.
The `simtools-run-science-tests` provides the runner tool to access and run the reusable tests
as defined in [simtools-tests](https://github.com/gammasim/simtools-tests).

The runner executes selected production, derivation, and comparison workflows from those definitions:

```text
science-test-template/
  catalogue.yml          test definitions and dependencies
  sites/*.yml            site templates
  workflows/             application workflows
  acceptance/            required products and numerical rules
  run_time.yml           optional shared container runtime
```

Each release has a `science_tests` directory containing `release.yml`, `context.yml`,
`sites/*.yml`, and generated `reports/`. The release selects required sites; each site file selects
its layout and required tests. The context provides the candidate and production-configuration
paths; a baseline path is required only for comparison tests.

## Setup and run

Create a release directory from the shared templates:

```console
simtools-run-science-tests --release_dir /path/to/release/science_tests --setup
```

This copies `release.yml`, every `sites/*.yml` template, and `context.yml`. Edit `context.yml` and
review the release and site settings. The runner uses `<release_dir>/context.yml` by default.

```console
# Validate definitions without execution.
simtools-run-science-tests --release_dir /path/to/release/science_tests --dry_run

# Generate and review the production grid.
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --test production.gamma.grid

# Submit production only after grid review.
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --test production.gamma --allow_production

# Check that production collection has completed.
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --test production.gamma.collect

# Explicitly derive products and compare them.
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --test derive.trigger_histograms \
  --test compare.trigger_histograms --test compare.compute_resources
```

Repeat `--site` or `--test` to narrow execution. Only named tests run; prerequisites must be selected
explicitly or their required products must already exist. Selected prerequisites run in dependency
order, and missing or incomplete inputs block dependents. Production submission returns `submitted`;
rerun `production.gamma.collect` to check completion. `--overwrite` archives a failed, finished
production for an explicit retry with `--allow_production`.

Expected products and acceptance rules are declared in the template. Missing products are
`incomplete`; failed numerical limits are `fail`; advisory limits are `warn` and require review.
Reports and the release summary are written under `reports/`; production data remains at the external
candidate path.
