# Science tests

Science tests are explicit, longer-running campaigns for physics performance, production behavior,
and resource comparisons. Typically, science tests are executed for each release of simtools.

## Definitions

Science-test definitions live in the [simtools-tests](https://github.com/gammasim/simtools-tests) repository:

```text
science-test-template/
  catalogue.yml                  test IDs, dependencies, gates, and sites
  workflows/                     production, derivation, and comparison workflows
  profiles/                      local and HTCondor settings
  acceptance/                    required products and acceptance rules

simtools-tests/<release>/science_tests/
  release.yml                    release label, baseline, sites, expected changes
  context.example.yml            external paths; copy outside Git
  sites/*.yml                    site layout and required tests
  acceptance/overrides.yml       must remain empty
  reports/                       small reports and collection inventories
```

The catalogue is reusable. Release files select the catalogue and sites; site files provide the
array layout and required tests. The external context provides the release label and absolute
candidate, baseline, and reference roots. Workflows use the application-workflow schema, while
the runner supplies site and report placeholders.

The template is discovered beside the release bundle or selected with `--template_dir`. A release
may override a workflow at the same relative path, but must provide an explicit catalogue.

## Run a campaign

Copy the release context example outside Git and set values such as:

```yaml
__SCIENCE_RELEASE_LABEL__: <release-label>
__SCIENCE_CANDIDATE_ROOT__: /data/science/candidate
__SCIENCE_BASELINE_ROOT__: /data/science/baseline
__SCIENCE_REFERENCE_ROOT__: /data/science/reference-inputs
```

The roots must be absolute; candidate and baseline must differ.

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

Use repeatable `--site` and `--test` options to select a subset; dependencies are added
automatically. Dry runs validate workflows and profiles without submitting jobs or writing reports,
but do not check external products.

Production consumers require a completed `submission.json` with non-empty job IDs, matching
expected-output entries, and existing output files. Existing submissions are never resubmitted
automatically. Failed or incomplete dependencies block downstream tests.

## Products and results

`acceptance/expected-products.yml` declares required non-empty products. `thresholds.yml` defines
named rules with JSON metrics and optional limits. Missing or invalid products/metrics are
`incomplete`; valid values outside a limit are `fail`. Advisory rules produce `warn` after checks
pass and still require scientific review. Only all-required `pass` results qualify a release.

Reports are written under the release `reports/` directory; large production data remains external.
The summary retains every required site/test, including `not_run`, `blocked`, and `incomplete`
entries. Retained results are reused only when the release, catalogue, sites, context, acceptance
rules, workflows, and profiles have the same fingerprint. Rerunning upstream evidence invalidates
dependent results.

| Status | Meaning |
| --- | --- |
| `planned` | Dry-run selection only. |
| `not_run` | Not executed or retained evidence was invalidated. |
| `blocked` | A dependency failed or was incomplete. |
| `incomplete` | Execution or required evidence failed. |
| `fail` | A valid measurement violated a configured limit. |
| `warn` | Checks passed; advisory review remains. |
| `pass` | Required products and configured checks passed. |

## Scope

The documented scope is the gamma-ray campaign. Automatic change-driven selection,
telescope-equivalence evidence, broader simulation-chain comparisons, waiver handling, baseline
promotion, and scientific approval workflows are outside that scope. A successful run does not
imply physics approval or baseline acceptance.

For implementation details, see [simtools-run-application](../user-guide/applications/simtools-run-application.md),
[simtools-compare-productions](../user-guide/applications/simtools-compare-productions.md), and
[simtools-run-science-tests](../user-guide/applications/simtools-run-science-tests.md).
