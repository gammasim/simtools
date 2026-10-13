# simtools-compare-science-results

```{eval-rst}
.. automodule:: simtools.applications.compare_science_results
   :members:
   :exclude-members: main
```

Compare two exported camera-efficiency summaries or manifests without running a calculation.
The JSON report records metric values, units, absolute and relative changes, compatibility
diagnostics, configured advisory tolerances, and plot paths. A zero baseline has no relative
change. Missing provenance or incompatible inputs produce an `incomplete` report.

Declare allowed case or input-hash changes with repeated `--compare_by` options. Set relative
advisory tolerances with repeated `--tolerance METRIC=VALUE` options. Wavelength tables are
compared only when both manifests contain them; unequal wavelength grids are diagnosed without
resampling.

```console
simtools-compare-science-results \
    --baseline_path /data/baseline/camera_efficiency_manifest.json \
    --candidate_path /data/candidate/camera_efficiency_manifest.json \
    --compare_by case.model_version \
    --tolerance cam_eff=0.05 \
    --output_path comparison
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: compare_science_results
   :no-heading:
```

## Integration example

```{eval-rst}
.. simtools-integration-example::
   :file: compare_science_results_camera_efficiency.yml
```
