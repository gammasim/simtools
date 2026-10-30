# simtools-plot-telescope-structure

Plot the optical geometry of a telescope, including its mirrors, camera housing,
and supporting structure. The geometry comes from model parameters and does not
include random alignment errors.

With `--shadowing`, the plot colours each structure element by the light it
stops on its own. This analysis requires the `obdeect` package, installed with
simtools' default dependencies.

## Example

```console
simtools-plot-telescope-structure --site North --telescope LSTN-01 \\
  --model_version 7.0.0 --output_path telescope-structure
```

To use a telescope structure definition from an ECSV file, pass
`--telescope_structure_file path/to/structure.ecsv`. This file overrides the
structure file selected by the telescope model.

Add `--shadowing` to include the shadowing analysis. Its star zenith angle,
photon count, and random seed can be set with `--zenith_angle`, `--photons`,
and `--seed`.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: plot_telescope_structure
   :no-heading:
```
