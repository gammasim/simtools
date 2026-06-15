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
comparison until a more complete production verification workflow is documented.

Example:

```bash
simtools-compare-productions \
    --config path/to/compare_productions.yml
```

Keep the comparison configuration and output reports with the production records so that later
checks can identify the exact baseline, production image, model version, and input files used.
