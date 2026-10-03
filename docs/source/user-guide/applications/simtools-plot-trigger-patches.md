# simtools-plot-trigger-patches

```{eval-rst}
.. automodule:: simtools.applications.plot_trigger_patches
   :members:
   :exclude-members: main
```

This application uses the selected camera model to generate LED events for
every sim_telarray trigger line. It passes those events through sim_telarray
and writes a PostScript camera display with `read_cta`.

The output directory retains the generated camera file, LED IACT input,
sim_telarray event file, PostScript display, and standard-output and
standard-error logs for each command. Use `--required_pixels` to select the
number of LEDs in each trigger combination. Use `--include_presum_slaves` when
the bracketed inputs of a pre-sum master should fire with that master.

Select one site, telescope, and model version. The sim_telarray installation
must contain `LightEmission/pixled`, `bin/sim_telarray`, and `bin/read_cta`.
For example:

```console
simtools-plot-trigger-patches --site North --telescope LSTN-01 \
  --model_version 7.0.0 --sim_telarray_path /path/to/sim_telarray \
  --output_path trigger-patches --required_pixels 2
```

The exported camera file is the input to `pixled --use-trg-all`. Each serialized
trigger line supplies its own eligible combinations; a table's `member_order`
does not represent a separate trigger group. In the model tables, `pixel_order`
orders a master followed by its pre-sum inputs, while `member_order` orders
the resulting tokens within the trigger line. A required majority master is
written as `+17`; a pre-sum is written as `17[18,19]`. Required pixels are
permitted only for majority triggers, and pre-sums only for majority or
digital-sum triggers. With `--include_presum_slaves`, pixels 18 and 19 fire
whenever master 17 fires. Slave firing is off by default.

The default output prefix is `trigger-patches-<telescope>-run<run>`, with
`.iact.gz`, `.simtel.gz`, and `.ps` artifacts and separate stdout/stderr logs
for `pixled`, `simtel`, and `read_cta`. The camera and matching model configuration
files are also retained. Use `--output_name inspection.ps` to change the display
filename. Open the `.ps` file with a PostScript viewer to inspect the mapping
consumed by sim_telarray.

The application stops at the first failed program or missing/empty output,
retaining intermediate files and logs for diagnosis. The display checks the
serialized mapping through the simulation software; it does not impose a
geometric adjacency or cross-patch reuse policy on camera designs.

```{eval-rst}
.. simtools-cli-help::
   :application: plot_trigger_patches
   :no-heading:
```
