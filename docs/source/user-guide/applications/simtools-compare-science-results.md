# simtools-compare-science-results

```{eval-rst}
.. automodule:: simtools.applications.compare_science_results
   :members:
   :exclude-members: main
```

Compare telescope-throughput results of two releases without running a calculation. Each input
is a combined summary written by `simtools-validate-telescope-throughput`, the directory
containing it, or a directory with one such result directory per telescope. Use `--telescopes`
to restrict the comparison to selected telescopes.

`science-comparison.json` records, per telescope, the summary metadata of both results and,
for every summary metric, the values in the baseline unit, the absolute and relative change, and
whether a configured advisory tolerance (`--tolerance METRIC=VALUE`) is exceeded. A zero
baseline has no relative change. Differing summary metadata, missing metrics, or tolerances for
unknown metrics make the comparison `incomplete`; allow metadata keys to differ with repeated
`--compare_by` options.

`report.pdf` contains, per telescope, the scalar changes, the detected spectra of all channels
with wavelength tables in both results (for example shower, muon, and NSB), and the optical
efficiency components of each channel (mirror reflectivity and shadowing, camera filter and pixel
throughput, quantum efficiency, atmospheric transmission, and total efficiency) as absolute
values and relative changes. Unequal wavelength grids are diagnosed without resampling. Table
problems are reported as `table_diagnostics`; with `--require_wavelength_tables` they make the
comparison `incomplete`.

```console
simtools-compare-science-results \
    --baseline_path /data/baseline/telescope_throughput \
    --candidate_path /data/candidate/telescope_throughput \
    --telescopes LSTN-01 LSTN-02 \
    --compare_by model_version \
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
   :file: compare_science_results_telescope_throughput.yml
```
