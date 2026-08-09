# Test resource benchmarks

Simtools records the resource requirements of its test suites in a separate
[benchmark dashboard](https://gammasim.github.io/simtools/dev/test-resources/).
The dashboard is updated by the scheduled and post-merge
`CI-test-resources` workflow. It contains the total unit-test suite and each
integration-test node, separated by model version.

The benchmark job runs serially on `ubuntu-latest` so measurements can be
attributed to one pytest invocation. It records wall time, CPU time, peak RSS
of pytest and its application descendants, and derived CPU utilisation. Hosted
runner measurements are noisy; use repeated runs to identify trends.

To investigate a change manually, dispatch `CI-test-resources` from the
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

The reason and excluded test ID are included in the benchmark run metadata.
Use this only for workloads that are not representative or cannot produce a
stable measurement.
