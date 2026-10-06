# Science tests

Science tests compare simulated event distributions and computing requirements against a reference
production. They typically run for each release of simtools.

## Definitions

Science-test definitions live in the [simtools-tests](https://github.com/gammasim/simtools-tests) repository:

```text
science-test-template/
  catalogue.yml                  test names, prerequisites, and sites
  workflows/                     production, derivation, and comparison workflows
  run_time.yml                   shared container runtime
  profiles/                      local and HTCondor settings
  acceptance/                    required products and acceptance rules

simtools-tests/<release>/science_tests/
  release.yml                    release label, baseline, sites, expected changes
  context.example.yml            external paths; copy outside Git
  sites/*.yml                    site layout and required tests
  acceptance/overrides.yml       must remain empty
  reports/                       comparison reports and lists of collected files
```

The catalogue is reusable. Release files select the catalogue and sites; site files provide the
array layout and required tests. The release file provides the release label. The external context
provides absolute candidate, baseline, and production-configuration directories. The candidate is
the production being tested;
the baseline is the production used for comparison. Workflows use the application-workflow schema;
the runner supplies site and report placeholders.

The template is discovered beside the release bundle or selected with `--template_dir`. A release
may override a workflow at the same relative path, but must provide an explicit catalogue.

The runner loads `run_time.yml` from the template directory for grid generation, derivation, and
comparison workflows. Its `runtime_environment` mapping uses the standalone runtime schema.
Placeholders are resolved from the campaign context; `__CONFIG_DIRECTORY__` refers to the directory
containing `run_time.yml`. Inline workflow runtime settings take precedence. Production submission
runs on the host, with containers configured in the HTCondor profile. Runtime settings are included
in the campaign signature.

## Run a campaign

Copy the release context example outside Git and set values such as:

```yaml
__SCIENCE_CANDIDATE_PATH__: /data/science/candidate
__SCIENCE_BASELINE_PATH__: /data/science/baseline
__PRODUCTION_CONFIGURATION_PATH__: /data/production-configuration/data
```

Use absolute directory paths and separate candidate and baseline directories.

First validate the complete selection:

```console
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --context_file /path/to/context.yml --dry_run
```

Then use the same release and context arguments to run the campaign:

```console
# Generate and review the grid.
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --context_file /path/to/context.yml --test production.gamma.grid

# Submit only after review; production is never implicit.
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --context_file /path/to/context.yml --test production.gamma --allow_production

# Derive and compare without resubmitting production.
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --context_file /path/to/context.yml \
  --test compare.trigger_histograms --test compare.compute_resources
```

Repeat `--site` and `--test` to select sites and tests; prerequisite tests are included
automatically. Dry runs validate workflows and profiles without submitting jobs or writing reports,
but do not check whether simulation outputs exist.

Pass `--simulation_models_git_path` and `--simulation_models_git_revision` on each runner command
to select the model repository used by its workflows.

Tests using simulation outputs require a completed `submission.json` with non-empty job IDs, matching
expected-output entries, and existing output files. Existing submissions are never resubmitted
automatically. Failed or incomplete prerequisite tests prevent dependent tests from running.

## Products and results

`acceptance/expected-products.yml` declares required non-empty products. `thresholds.yml` defines
named rules with measurements read from JSON files and optional limits. Missing or invalid files
or measurements are `incomplete`; valid values outside a limit are `fail`. Advisory rules produce
`warn` after checks
pass and still require scientific review. Only all-required `pass` results qualify a release.

Reports are written under the release `reports/` directory; large production data remains external.
The summary includes every required test at each site, including tests that did not run or could
not complete. Results are reused only when the release, catalogue, site settings, context,
acceptance rules, workflows, and profiles are unchanged. Rerunning a prerequisite test requires
its dependent comparisons to be repeated.

| Status | Meaning |
| --- | --- |
| `planned` | Dry-run selection only. |
| `not_run` | Not executed or needs to be repeated. |
| `blocked` | A prerequisite test failed or was incomplete. |
| `incomplete` | Execution failed or required outputs were missing or invalid. |
| `fail` | A valid measurement violated a configured limit. |
| `warn` | Checks passed; scientific review is required. |
| `pass` | Required products and configured checks passed. |

## Scope

The documented scope is gamma-ray simulations. Automatic test selection based on configuration
changes, combining results from equivalent telescopes, broader simulation-chain comparisons,
exceptions to acceptance limits, selecting new reference productions, and formal scientific
approval are outside that scope. A successful run does not imply physics approval or acceptance
of a reference production.

For implementation details, see [simtools-run-application](../user-guide/applications/simtools-run-application.md),
[simtools-compare-productions](../user-guide/applications/simtools-compare-productions.md), and
[simtools-run-science-tests](../user-guide/applications/simtools-run-science-tests.md).
