# Science tests

Science tests run configured production, derivation, and comparison workflows for a release. Shared
definitions are in [simtools-tests](https://github.com/gammasim/simtools-tests):

```text
science-test-template/
  catalogue.yml          test definitions and dependencies
  sites/*.yml            site templates
  workflows/             application workflows
  acceptance/            required products and numerical rules
  run_time.yml           optional shared container runtime
```

Each release has a small `science_tests` directory containing `release.yml`, `context.yml`,
`sites/*.yml`, and generated `reports/`. The release selects required sites; each site file selects
its layout and required tests. The context provides absolute candidate, baseline, and
production-configuration paths.

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

# Collect production, derive products, and compare them.
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --test compare.trigger_histograms --test compare.compute_resources
```

Repeat `--site` or `--test` to narrow execution; dependencies are included automatically. Production
submission returns `submitted`; rerun `production.gamma.collect` to check completion. Failed or
incomplete prerequisites block dependents. `--overwrite` archives a failed, finished production for
an explicit retry with `--allow_production`.

Expected products and acceptance rules are declared in the template. Missing products are
`incomplete`; failed numerical limits are `fail`; advisory limits are `warn` and require review.
Reports and the release summary are written under `reports/`; production data remains at the external
candidate path.
