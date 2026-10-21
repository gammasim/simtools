# simtools-plot-trigger-patches

`simtools-plot-trigger-patches` creates a static report from a telescope model's
camera geometry and trigger-patch membership. It does not run `pixled`,
sim_telarray, or CORSIKA, and it does not test trigger firing or thresholds.

The first PDF page is the existing **patch configuration** map, colored by the
number of configured patches containing each camera pixel. The second page maps
each pixel's median physical distance to the other members of each patch it
belongs to, then takes the median across its patches. Distances use the camera
model's coordinate units (cm). Pixels with no patch membership or no peer in a
patch are shown as undefined. Their `distance_status` in the pixel table
distinguishes `no patch membership` from `no other members in patch`; uncovered
pixel IDs also appear in the page summary.

Group summaries include patch size, median and maximum pairwise distances, and
connected-component counts based on the camera's neighbor geometry. These are
screening diagnostics: patch layouts can differ by camera, so the report does
not apply a universal distance threshold or decide that uncovered pixels are
misconfigured. Trigger eligibility needs an explicit design/model role.
The page-two panel lists only groups with an unusually large maximum span
(above the Tukey upper fence, calculated within each group kind) or with
disconnected components. Distances are rounded to the displayed precision
before outlier detection, so harmless floating-point variation does not create
extra entries. These entries are review candidates, not automatic failures.

## Example

```console
simtools-plot-trigger-patches --site North --telescope LSTN-design \
  --model_version 7.0.0 --output_path trigger-patch-report
```

The output directory contains a two-page PDF, per-pixel and per-group ECSV
tables, and a JSON summary. Both pages place the camera map at the same position;
page one reserves the summary-panel area used on page two. The page-two group
and coverage summary is also written to the application log for terminal review.
Both places list every pixel ID for each group shown; the ECSV tables retain the
full configuration. The JSON includes model identifiers, a configuration
digest, coverage status, and a runtime-trigger status marked `not tested`. A
`pass` membership coverage status means every camera pixel appears in at least
one patch; it does not validate firing or trigger eligibility. Use
`--output_name` to set the shared file-name base.

```{eval-rst}
.. automodule:: simtools.applications.plot_trigger_patches
   :members:
   :exclude-members: main
```

```{eval-rst}
.. simtools-cli-help::
   :application: plot_trigger_patches
   :no-heading:

.. simtools-integration-example::
    :file: plot_trigger_patches.yml
```
