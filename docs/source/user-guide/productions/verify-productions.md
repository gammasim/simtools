# Verify Productions

```{warning}
Production verification documentation is incomplete. The checks below describe the current
simtools tools and expected workflow, but production-specific acceptance criteria still need to
be defined.
```

Verification starts during execution.
[simtools-simulate-prod](../applications/simtools-simulate-prod) validates the generated
simulation outputs, reports the run status, and can write lists of output and log files with
`--save_file_lists`.

Production verification should include:

- checking CORSIKA, sim_telarray, and simtools log files for failed runs
- checking that expected eventio, histogram, reduced-event, and registration files exist
- verifying sim_telarray meta-parameters and generated configuration values
- comparing selected output distributions with a baseline production
- documenting failed jobs, resubmissions, and accepted deviations

Use [simtools-compare-productions](../applications/simtools-compare-productions) to compare a
production with a baseline production. This is the main simtools application for production
comparison until a more complete production verification workflow is documented. Event-level
comparison uses trigger histogram HDF5 files produced with
[simtools-write-trigger-histograms](../applications/simtools-write-trigger-histograms).

Example:

```bash
simtools-compare-productions \
    --config path/to/compare_productions.yml
```

Keep the comparison configuration and output reports with the production records so that later
checks can identify the exact baseline, production image, model version, and input files used.

Event-level comparison writes `comparison_statistics.json` alongside the plots. The report uses the
KS statistic for continuous quantities, Jensen-Shannon distance for categorical trigger and
telescope distributions, and Wasserstein-1 distance for ordered trigger multiplicities. Distances
compare normalized distribution shapes, so differing production event totals do not directly set
the result. A `comparison_statistics.meta.yml` sidecar records the application configuration and
provenance. The report format is described by
`src/simtools/schemas/production_comparison_statistics.schema.yml`. The report file contains
diagnostics; simtools does not apply acceptance thresholds.

For telescope-level diagnostics, use `--comparison_level signal` with one
`--array_layout_name` and sim_telarray files as the production inputs. The application discovers
the telescopes in the input layout and writes pedestal, integrated-signal, peak-sample, and
triggered-pixel distributions under one output directory per telescope. If an input does not
contain a triggered-pixel list, the selected-pixel list is used instead.
Signal and timing distributions use the KS statistic; triggered-pixel multiplicities use the
Wasserstein-1 distance, matching the event-level comparison convention.
