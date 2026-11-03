# simtools-derive-bias-curves

```{eval-rst}
.. automodule:: simtools.applications.derive_bias_curves
   :members:
   :exclude-members: main
```

## Description

This application combines NSB (Night Sky Background) and proton trigger rates
to generate bias curves showing how trigger rates vary with threshold.

The tool:
1. Extracts NSB trigger rates from reduced event-data HDF5 files
2. Calculates proton trigger rates from proton reduced event-data HDF5 files
3. Plots both curves on the same figure for comparison
4. Outputs ECSV tables for runwise NSB simulation, runwise proton simulation, and combined bias curves
5. Calculates the trigger threshold as the intersection point between the NSB curve and the scaled proton curve
6. Exports the trigger threshold as a model parameter (`asum_threshold`, `dsum_threshold`, or `discriminator_threshold` depending on the telescope's default trigger type)

The input directory should contain both:
- NSB reduced event-data HDF5 files (e.g., `gamma*.reduced_event_data.hdf5`)
- Proton simulation reduced event-data HDF5 files (e.g., `proton*.reduced_event_data.hdf5`)

The input files can be generated using `simtools-generate-bias-curve-submissions`.

### Notes

- The trigger threshold is the intersection of the log-linear fits shown on the plot
  for NSB and proton rates scaled by `0.1 * scaling_factor`. Each fit uses its finite,
  positive rates and requires at least two distinct thresholds. The intersection
  must fall within the threshold range shared by both fits.
- Thresholds are read from per-telescope values in HDF5 `SIMULATION_METADATA`.
  `default_trigger` selects `asum_threshold`, `dsum_threshold`, or
  `discriminator_threshold`. Decimal values remain separate simulation points.
  All telescopes and inputs in a file must share the same active parameter and
  threshold. Files with missing, invalid, or inconsistent threshold metadata are
  skipped; thresholds are not inferred from filenames.
- The exported model parameter (`asum_threshold`, `dsum_threshold`, or `discriminator_threshold`) is written to the standard model data output directory under `<telescope>/<parameter_name>/`.
- If no intersection point is found, the application raises an error.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: derive_bias_curves
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: derive_bias_curves_mstn-01.yml
```
