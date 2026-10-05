# Science tests

Science tests are explicit, longer-running campaigns for physics performance, production
behavior, and resource comparisons. They are not part of the default pull-request test layer.

Execution evidence is not scientific acceptance. Pin all inputs except the component under test,
make comparison differences explicit, and use enough events to support the conclusion. The
current runner implements the gamma pilot; other science-test families are not registered yet.

## Campaign structure

Definitions live in the `simtools-tests` repository:

```text
science-test-template/
  catalogue.yml                  reusable test definitions and dependencies
  workflows/                     production, derivation, and comparison workflows
  profiles/                       local and HTCondor settings
  acceptance/                    required products and acceptance rules

simtools-tests/<release>/science_tests/
  release.yml                    release label, baseline, sites, expected changes
  context.example.yml            external paths; copy outside Git
  sites/north.yml, south.yml     site layout and required tests
  acceptance/overrides.yml       currently must remain empty
  reports/                       small reports and collection inventories
```

The catalogue defines stable test IDs, supported sites, dependencies, tiers, input gates, and
acceptance rules. Site files select the tests required for each site and provide the array layout.
The external context supplies the release label and absolute candidate, baseline, and reference
roots. Workflows use the existing application-workflow schema; the runner supplies site and report
placeholders.

The template is discovered beside the release bundle or selected with `--template_dir`. A release
may override a workflow at the same relative path, but must still provide an explicit catalogue.

## Current pilot

The v0.38.0 pilot runs the same catalogue for North and South:

| Test | Purpose | Gate |
| --- | --- | --- |
| `production.gamma.grid` | Generate a grid for review. | Explicit selection. |
| `production.gamma` | Submit and wait for production. | Reviewed grid and `--allow_production`. |
| `derive.trigger_histograms` | Derive candidate histograms. | Completed production. |
| `compare.trigger_histograms` | Compare event histograms. | Candidate derivation. |
| `compare.compute_resources` | Compare resource requirements. | Completed production. |

The pilot compares candidate `v0.38.0_rc1` with baseline `v0.37.1` and declares
`configuration.atmosphere` as the expected difference. These are campaign settings, not reusable
defaults or approval of a numerical change.

## Run a campaign

Copy the release context example outside Git and fill in values like:

```yaml
__SCIENCE_RELEASE_LABEL__: v0.38.0_rc1
__SCIENCE_CANDIDATE_ROOT__: /data/science/v0.38.0_rc1
__SCIENCE_BASELINE_ROOT__: /data/science/approved-v0.37.1
__SCIENCE_REFERENCE_ROOT__: /data/science/reference-inputs
```

The roots must be absolute, and candidate and baseline must differ. Keep large products, logs, and
execution provenance under the candidate root.

Preflight the complete selection:

```console
simtools-run-science-tests --release_dir /path/to/release/science_tests \
  --context_file /path/to/context.yml --dry_run
```

Then use the same release and context arguments for the campaign:

```console
# Generate and review grids.
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

Use repeatable `--site` and `--test` options to select a subset. Dependencies are included
automatically. Dry runs validate workflows and profiles without submitting jobs or writing reports;
they do not check whether external products already exist.

Production consumers require a completed `submission.json` with non-empty job IDs, matching
expected-output entries, and existing output files. An existing submission is never resubmitted
automatically. Failed or incomplete dependencies block downstream tests.

## Products and results

`acceptance/expected-products.yml` declares required non-empty products. `thresholds.yml` defines
named rules with JSON metrics and optional minimum/maximum limits. Missing or invalid products and
metrics are incomplete; valid values outside a limit fail. `mode: advisory` produces `warn` after
the checks pass and still requires scientific review. Only all-required `pass` results qualify a
release; the pilot's numerical thresholds are completeness checks, not calibrated physics limits.

Reports are written to the release bundle under `reports/`, while large production data remains
external. The summary retains the complete required-site matrix, including `not_run`, `blocked`,
and `incomplete` tests. Subset results are reused only when the release, catalogue, site files,
context, acceptance rules, workflows, and profiles have the same fingerprint; rerunning upstream
evidence invalidates dependent results.

Typical statuses are:

| Status | Meaning |
| --- | --- |
| `planned` | Dry-run selection only. |
| `not_run` | Required test was not executed or its retained evidence was invalidated. |
| `blocked` | A dependency failed or was incomplete. |
| `incomplete` | Execution or required evidence failed. |
| `fail` | A valid measurement violated a configured limit. |
| `warn` | Checks passed, but advisory scientific review remains. |
| `pass` | Required products and configured checks passed. |

## Boundaries and related docs

The runner currently archives the gamma pilot. Automatic change-driven selection, telescope
equivalence, broader simulation-chain comparisons, waiver handling, baseline promotion, and
scientific approval workflows are not implemented. A successful run therefore does not imply
physics approval or baseline acceptance.

For implementation details, see:

- [simtools-run-application](../user-guide/applications/simtools-run-application.md)
- [simtools-compare-productions](../user-guide/applications/simtools-compare-productions.md)
- [simtools-run-science-tests](../user-guide/applications/simtools-run-science-tests.md)
