# simtools-plot-trigger-patches

```{eval-rst}
.. automodule:: simtools.applications.plot_trigger_patches
   :members:
   :exclude-members: main
```

This application uses the selected camera model to generate LED events for
sim_telarray trigger groups and writes a compact inspection report. It compares
resolved model membership and roles with the exported camera, then compares
membership and trigger kind with the simulator header. Raw simulator thresholds
are retained with their normalization convention; they are not compared directly
with readout ADC integrals.

By default, each group receives one whole-group illumination pattern. Identical
patterns are simulated once, retaining every source-group association. This
replaces the branch's former default of enumerating all two-pixel combinations.
Use `--required_pixels` for a specific master count and `--exhaustive` to enumerate
every combination of that explicitly selected size. Shared patterns produce
correlated observations across overlapping groups.

For a quick central-camera inspection, cap the total trials and select a radius:

```console
simtools-plot-trigger-patches --site North --telescope LSTN-01 \
  --model_version 7.0.0 --output_path trigger-patches \
  --max_events 5 --max_pixel_radius 0.2
```

The event cap includes repetitions and applies before photon generation. Radius
selection includes pre-sum slaves when enabled. Whole groups outside the radius
are not illuminated; all groups still receive structural comparisons and explicit
coverage accounting. Limited runs use deterministic ordering, not random sampling.

The PDF contains two summary pages and at most four diagnostic events. Complete
per-event and per-group results are retained as `.events.ecsv` and `.groups.ecsv`
beside the PDF, with provenance and reconciled counts in `.summary.json`. Missing
events, duplicate IDs, header/model inconsistencies, and displaced photoelectrons
cause an inspection failure after the report is saved. A non-triggering trial is
retained separately from a missing record.

For single-mirror telescopes, the separate `.illumination.dat` file compensates
the rotation and y-axis flip that sim_telarray still applies with optics bypassed.
The original model camera is retained and is used by the simulator. Arrival-pixel
checks also expose unsupported geometry conventions in other camera designs.

Each diagnostic page shows a recorded readout image when available. Otherwise it
shows the Monte Carlo photoelectron counts and explicitly states that no waveform
was recorded. Cyan rings mark
the pixel IDs emitted by `pixled`, orange crosses mark pixels with Monte Carlo
photoelectrons, blue outlines mark configured groups containing the complete
input combination, and red outlines mark pixels in sectors reported as fired by
sim_telarray. A panel lists the `pixled` input pixels, their configured sector
membership, any sector containing the full input combination, the photoelectron
pixels, and the fired sectors. One input pair can fire several overlapping
sectors. Off-target photoelectron arrivals are flagged as mapping mismatches.

The `pixled` input is retained in an event manifest before photon generation,
matched by the event's shower number. Older branch artifacts can be inspected by
recovering their inputs from the retained pixled log, with unavailable original
model comparisons labeled explicitly. Photoelectron pixels come
from the sim_telarray event data. Fired sector IDs come from the event header
and index trigger lines in the retained camera file, starting at zero.
Diagnostic page numbers are not input-combination numbers; event IDs are printed
in the title. Every simulated
trial is accounted for, including trials without a telescope trigger. The heatmap
shows integrated readout signals, so brightness is not itself the trigger
decision. Adjacent pixels with similar signals can look like a single bright
patch; the markers and IDs identify the two stages explicitly.

`--required_pixels` selects the number of illuminated master pixels in a stimulus.
Omitting it illuminates every master of each selected group.
`trigger_pixels` is a model parameter controlling majority-trigger multiplicity;
explicit group multiplicities and required masters also affect that logic.
Analog and digital sum triggers use their sum thresholds instead. Setting the
stimulus size to the majority multiplicity therefore does not establish a
threshold test, especially for cameras using sum triggers.

The photon intensity is an input setting, not a calibrated trigger operating
point. Functional acceptance is reported as inconclusive: a non-trigger alone
does not prove a missed expected trigger. Establish below/above-threshold light
levels and repetitions before applying physical pass/fail criteria.

Select one site, telescope, and model version. The sim_telarray installation
must contain `LightEmission/pixled` and `bin/sim_telarray`.
PDF generation uses Matplotlib within simtools. The atmospheric profile is
exported from the selected site and model version, converted to native format
when needed, and copied to `atmprof1.dat` for `pixled --atmosphere 1`.
For example:

```console
simtools-plot-trigger-patches --site North --telescope LSTN-01 \
  --model_version 7.0.0 \
  --output_path trigger-patches --required_pixels 2
```

The original camera supplies the trigger definitions; pixled reads exact planned
patterns from the coordinate-corrected illumination copy. A table's `member_order`
does not represent a separate trigger group. In the model tables, `pixel_order`
orders a master followed by its pre-sum inputs, while `member_order` orders
the resulting tokens within the trigger line. A required majority master is
written as `+17`; a pre-sum is written as `17[18,19]`. Required pixels are
permitted only for majority triggers, and pre-sums only for majority or
digital-sum triggers. With `--include_presum_slaves`, pixels 18 and 19 fire
whenever master 17 fires. Slave firing is off by default.

An explicit `--required_pixels` must fit every configured group; smaller groups
are rejected before generation. Required majority masters are always included,
even if their count exceeds the requested stimulus size. Pre-sum slaves do not
count as additional masters. A camera without trigger groups is rejected.

The default output prefix is `trigger-patches-<telescope>-run<run>`, with
`.iact.gz`, `.simtel.gz`, and `.pdf` artifacts and separate stdout/stderr logs
for `pixled` and `simtel`. The camera and matching model configuration
files are also retained. Use `--output_name inspection.pdf` to change the display
filename. Open the `.pdf` file with a PDF viewer to inspect the mapping
consumed by sim_telarray.

The application retains intermediate files and logs for diagnosis. Camera-specific
adjacency and pixel-reuse policies require an approved design reference.

```{eval-rst}
.. simtools-cli-help::
   :application: plot_trigger_patches
   :no-heading:
```
