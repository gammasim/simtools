# Test resource benchmarks

Simtools records the resource requirements of its test suites in a separate
[benchmark dashboard](https://gammasim.github.io/simtools/dev/test-resources/).
The dashboard is updated by the scheduled and post-merge
`CI-test-benchmarks` workflow. It contains the unit and integration suite totals
and the slower integration-test nodes, separated by model version.

Each benchmark suite runs once and serially on `ubuntu-latest`. A pytest plugin
records wall time, CPU time, peak RSS of pytest and its application descendants,
on hosted runners. These measurements are noisy; use repeated runs to identify
trends.

Individual integration-test charts are published only when the test wall time
meets the configurable threshold, which defaults to five seconds. Fast tests
still run, contribute to the suite totals, and remain available in the raw
workflow artifact. Change `integration_min_wall_time` when manually dispatching
the workflow to investigate a different range.

To investigate a change manually, dispatch `CI-test-benchmarks` from the
Actions page and provide the model-version list. Pull requests do not publish
benchmark data. Existing correctness workflows remain the source of test
status.

## Excluding an integration test

An application workflow can exclude one test from resource benchmarking while
leaving normal integration CI unchanged. Set a non-empty reason in the YAML
configuration:

```yaml
applications:
- application: simtools-example
  exclude_from_resource_benchmark: "Requires an unstable external service."
  configuration:
    output_path: simtools-output
```

The reason and excluded test ID are included in the raw benchmark artifact.
Use this only for workloads that are not representative or cannot produce a
stable measurement.
