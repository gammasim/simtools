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
optional `sites/*.yml` overrides, and generated `reports/`. The release selects required sites;
shared site templates supply defaults. The context provides the candidate and production-configuration
paths; a baseline path is required only for comparison tests.

## Setup and run

Create a release directory from the shared templates:

```console
simtools-run-science-tests --release_dir /path/to/release/science_tests --setup \
  --baseline_version v0.38.0 --simulation_models_git_revision v0.18.0
```

This copies `release.yml`, every `sites/*.yml` template, and `context.yml`. The optional setup
values set the baseline and model ref in `release.yml`, and the baseline directory in `context.yml`.
Edit `context.yml` and
review the release and site settings. The runner uses `<release_dir>/context.yml` by default.

Set an absolute shared storage root in your local environment before running:

```console
export SIMTOOLS_SCIENCE_TESTS_ROOT=/path/to/shared/science_tests
```

Alternatively, set `SIMTOOLS_SCIENCE_TESTS_ROOT=/path/to/shared/science_tests` in an untracked `.env`
in the working directory, or use `--env_file` to select a different environment file. Keep the
filesystem-specific root in local configuration and reference it in the committed context:

```yaml
__SCIENCE_CANDIDATE_PATH__: ${SIMTOOLS_SCIENCE_TESTS_ROOT}/vX.Y.Z
__SCIENCE_BASELINE_PATH__: ${SIMTOOLS_SCIENCE_TESTS_ROOT}/vA.B.C
__PRODUCTION_CONFIGURATION_PATH__: ${SIMTOOLS_SCIENCE_TESTS_ROOT}/production-configuration/data
__SCIENCE_CONTAINER_IMAGE_PATH__: ${SIMTOOLS_SCIENCE_TESTS_ROOT}/vX.Y.Z/simtools-vX.Y.Z.sif
__CORSIKA_INTERACTION_TABLE_PATH__: ${SIMTOOLS_SCIENCE_TESTS_ROOT}/corsika7-interaction-tables/interaction-tables
simulation_models_git_path: ${SIMTOOLS_SCIENCE_TESTS_ROOT}/simulation-models
```

Replace the release versions and image filename with the intended candidate and baseline. For a
checked-out model repository, use `simulation_models_path` instead of `simulation_models_git_path`.
The runner resolves `${SIMTOOLS_SCIENCE_TESTS_ROOT}` when loading the context, before path
validation, preflight, and workflow preparation. An unset, empty, or relative root is an error when
referenced; contexts using explicit absolute paths remain supported. The effective workflows
record the resolved paths. All paths must be accessible at the same location on the submit host
and inside Apptainer on the workers; use `--preflight` to check this before production submission.

```console
# Validate definitions without execution.
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --context_file /path/to/release/science_tests/context.yml --dry_run

# Generate and review the production grid.
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --context_file /path/to/release/science_tests/context.yml --test production.gamma.grid

# Submit production only after grid review.
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --context_file /path/to/release/science_tests/context.yml \
  --test production.gamma --allow_production --preflight

# Collect finished jobs, validate their products, and update the release report.
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --context_file /path/to/release/science_tests/context.yml \
  --test production.gamma.collect

# Explicitly derive products and compare them.
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --context_file /path/to/release/science_tests/context.yml \
  --test derive.trigger_histograms \
  --test compare.trigger_histograms --test compare.compute_resources
```

`--preflight` is recommended before production. It validates all configured host paths, then runs
one high-priority worker-side Apptainer canary per selected site using the production container
setup. If omitted, production still proceeds with a warning.

Repeat `--site` or `--test` to narrow execution. Only named tests run; prerequisites must be selected
explicitly or their required products must already exist. Selected prerequisites run in dependency
order, and missing or incomplete inputs block dependents. Production submission returns `submitted`.
`production.gamma.collect` checks submitted jobs, collects finished jobs, validates the required
products, and records whether collection is pending or complete in the release summary. Rerun it
while jobs remain pending. `--overwrite` archives a failed, finished production for an explicit
retry with `--allow_production`.

Telescope-throughput site definitions can list several telescopes in `camera_telescopes`:

```yaml
camera_telescopes: [LSTN-01, LSTN-02]
```

Run `telescope_throughput` separately for each release, using that release's science-test directory,
context, container image, and model source. Each run writes telescope results and manifests under
its configured release data directory. For example, select a site and run:

```console
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --context_file /path/to/release/science_tests/context.yml --site north \
  --test telescope_throughput
```

Once both releases have results, configure the baseline and candidate paths in the candidate
release context and run `--test compare.telescope_throughput`. The comparison reads the existing
results and publishes `report.pdf` and `science-comparison.json` to
`reports/<site>/compare.telescope_throughput/` in that release directory. Its analysis image must
contain `simtools-compare-science-results`. Execution logs and effective workflows remain below
`$SIMTOOLS_SCIENCE_TESTS_ROOT/<release>/work/science-tests/`.

Expected products and acceptance rules are declared in the template. Missing products are
`incomplete`; numerical acceptance breaches and advisory rules are `warn` and require review.
Warnings do not qualify a release; every required test must pass.
Missing configured comparison metrics produce warnings; duplicate measurements are errors.
All acceptance deviations are collected in a small `acceptance.md` report linked from the
release summary, with PDF page links where available. Complete scalar results remain in CSV.
The release summary and small acceptance reports are written under `reports/`; production data and
science-test artifacts remain under the external candidate or baseline paths.
Summary generation does not scan candidate or baseline data. If those files change outside the
runner, rerun the affected tests to refresh their results.

The runner records effective execution settings with each external workflow.
