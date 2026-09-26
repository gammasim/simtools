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

```{eval-rst}
.. simtools-cli-help::
   :application: plot_trigger_patches
   :no-heading:
```
