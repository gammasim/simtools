# simtools-compare-science-results

```{eval-rst}
.. automodule:: simtools.applications.compare_science_results
   :members:
   :exclude-members: main
```

Compare exported camera-efficiency results without running a calculation. Each input may be a
manifest, a legacy summary, or a directory containing one manifest per telescope. Use
`--telescopes` to select telescope names from those directories.
The JSON report records metric values, units, absolute and relative changes, compatibility
diagnostics, configured advisory tolerances, plot paths, and the case, simtools version, and
image identity for each result. A zero baseline has no relative change. Missing provenance or
incompatible inputs produce an `incomplete` report.

Declare allowed case or input-hash changes with repeated `--compare_by` options. Set relative
advisory tolerances with repeated `--tolerance METRIC=VALUE` options. When all three wavelength
tables are available, the report includes shower, muon, and NSB absolute spectra and relative
candidate-to-baseline changes, using the detected-light columns from each table. When only
optical throughput is available, the panel is labeled accordingly. It also compares mirror reflectivity and shadowing, camera filter
and pixel throughput, photodetector quantum efficiency, atmospheric transmission, and total
telescope efficiency. Component plots show absolute values and relative candidate-to-baseline
changes. Unequal wavelength grids are diagnosed without resampling. All comparison plots are
collected into `report.pdf`; the application does not write separate PNG files.

```console
simtools-compare-science-results \
    --baseline_path /data/baseline/camera_efficiency \
    --candidate_path /data/candidate/camera_efficiency \
    --telescopes LSTN-01 LSTN-02 \
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
   :file: compare_science_results_telescope_throughput.yml
```
