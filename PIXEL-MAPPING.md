# Trigger-patch mapping with sim_telarray

Implement [issue #2539](https://github.com/gammasim/simtools/issues/2539): a
simtools application that turns every serialized camera trigger definition into
LED events with sim_telarray's `pixled`, processes those events with
`sim_telarray`, and writes the camera view produced by `read_cta`.

The output is an executable check of the mapping from the model-table trigger
definitions to sim_telarray camera pixels. It is not a new geometric validator
or a hand-drawn representation of the table rows.

## Verified behaviour to preserve

`LightEmission/pixled_count.sh` is useful for its invocation, but only counts
combinations. `LightEmission/pixled.cc` establishes the behaviour required by
this issue:

1. `pixled --camera-file <camera.dat> --use-trg-all` reads every line containing
   `Trigger` from the generated camera file, one line at a time.
2. For each trigger line it cleans the sim_telarray trigger syntax and generates
   LED events for every eligible combination. It does not operate on a model
   table row or label a `member_order` as a separate trigger group.
3. `--require N` selects combinations with `N` fired pixels. With no value, all
   pixels selected from that trigger line fire. A `+` marker in the trigger line
   makes that pixel mandatory for the generated combinations.
4. `--slaves` makes the pixels inside `master[slave,...]` fire whenever their
   master fires. The first implementation must expose this as an explicit
   option, defaulting to the issue's example behaviour (off).
5. `pixled` produces an IACT input stream. `sim_telarray` consumes that stream;
   `read_cta -p` produces the PostScript camera display. The display is the
   authoritative visual result because it uses sim_telarray's trigger parser,
   camera geometry, electronics, and event format.

## Current branch state

The committed branch work already provides the model-table path needed by this
application:

- `camera_trigger_groups` and `camera_trigger_members` are resolved by
  `TelescopeModel` and written by `simtools.simtel.simtel_file_writer` as
  `MajorityTrigger`, `AnalogSumTrigger`, or `DigitalSumTrigger` lines.
- Commit `fcb8e3aef` registers `simtools-plot-trigger-patches` and adds its
  documentation references. Keep that public command name.
- The current worktree has untracked files named `plot_trigger_patches` and
  `trigger_patch_validator`. They draw a PDF from `Camera.from_configuration`
  and derive adjacency from polygons. They do not run `pixled`, `sim_telarray`,
  or `read_cta`, so they do not satisfy issue #2539. Do not build on the
  adjacency validator or retain its PDF/JSON output contract for this issue.

Before adding the application, correct the serializer checks that the existing
tables depend on. In particular, reject `required: true` on any non-master row
(the current condition misses that case if the master is not required), reject
`+` for non-majority triggers, and reject bracket syntax for analog-sum
triggers. Retain the existing contiguous IDs/orders, known-pixel, nonempty
group, and multiplicity checks. These checks make failures clear before an
external program is run; `pixled` and sim_telarray remain the integration
authority.

## Implementation plan

### 1. Replace the untracked prototype with a runner-focused library module

Create `src/simtools/simtel/trigger_patch_mapping.py`. Keep it independent of
CLI parsing and make it accept resolved model objects and explicit paths.

Implement one public orchestration function, for example
`run_trigger_patch_mapping(telescope_model, site_model, output_directory, ...)`.
It must:

1. Call the existing model export path to write the camera file and the matching
   telescope/site sim_telarray configuration into the application output area.
   Pass the exact generated camera file to `pixled`; do not recreate trigger
   lines from the ECSV tables in this module.
2. Locate these executables from `settings.config.sim_telarray_path`:
   `LightEmission/pixled`, `bin/sim_telarray` (via
   `settings.config.sim_telarray_exe`), and `bin/read_cta`. Fail early with the
   missing full path in the exception.
3. Run `pixled` with a list-form command, never a shell string:
   `pixled --camera-file <generated-camera-file> --use-trg-all --require <N>`.
   Add `--slaves` only when requested. Set `--photons`, `--events`, `--run`, and
   `-o <iact-output>` from explicit function arguments.
4. Run `sim_telarray` on the IACT stream. Start from the exported configuration
   and include its config directory using `-I`. Set the output file, a bounded
   I/O buffer, and the minimum overrides required for an LED camera event:
   `Bypass_Optics=2`, `maximum_telescopes=1`, telescope pointing `(0, 0)`, and
   the selected site altitude and atmospheric transmission. Derive altitude and
   atmosphere from `SiteModel`; do not hard-code the example's 2150 m or
   atmosphere filename. Confirm the final option names against the installed
   sim_telarray version when implementing.
5. Run `read_cta -p <postscript-output> <simtel-output>`. Return a dataclass
   containing the camera file, pixled IACT stream, simtel event file,
   PostScript file, and stdout/stderr log files. Preserve all of them in the
   output directory for review.

Use `job_manager.submit` with list commands and named stdout/stderr files, in
the same style as existing sim_telarray runners. Do not introduce a shell
pipeline or use `SimulatorLightEmission`: it assumes a calibration source and
model parameters that `pixled` does not use.

### 2. Add the application around the runner

Replace the untracked `src/simtools/applications/plot_trigger_patches.py` with
an application that creates `TelescopeModel` and `SiteModel`, then calls the
runner. It needs `database=True` and these arguments:

- required model selection: `--site`, `--telescope`, and `--model_version`;
- `--sim_telarray_path` and `--output_path`;
- `--required_pixels` (positive integer, default `2`), mapped to
  `pixled --require`;
- `--include_presum_slaves` (flag), mapped to `pixled --slaves`;
- `--photons_per_pixel`, `--events_per_combination`, and `--run_number`, with
  pixled-compatible defaults;
- `--output_name` for the PostScript basename, with a deterministic default.

Keep `simtools-plot-trigger-patches` because it is already registered in
`pyproject.toml` and listed in the applications page. Its completion log must
name the PostScript output and the generated simtel event file. Do not expose
the prototype's `--adjacency_tolerance`, `--strict`, `--show_pixel_ids`, or
`--hide_member_order` options.

### 3. Make output names and failure handling reviewable

Use one stable prefix based on telescope name and run number, for example:

```text
trigger-patches-<telescope>-run<run>.camera.dat
trigger-patches-<telescope>-run<run>.iact.gz
trigger-patches-<telescope>-run<run>.simtel.gz
trigger-patches-<telescope>-run<run>.ps
trigger-patches-<telescope>-run<run>.<step>.stdout.log
trigger-patches-<telescope>-run<run>.<step>.stderr.log
```

Stop at the failing program. Let `job_manager` surface its log excerpt and add
the generated file paths to the raised error where available. Do not parse
`read_cta` output to infer patch membership; the visual artifact and the source
camera file are the deliverables.

### 4. Tests

Read `.agents/skills/unit-testing/SKILL.md` before adding the unit tests.

Add `tests/unit_tests/simtel/test_trigger_patch_mapping.py` for the library
module. Mock `settings.config.sim_telarray_path`, model export, and
`job_manager.submit`; assert the three commands, their order, the camera file
used by `pixled`, conditional `--slaves`, site-derived sim_telarray values,
and deterministic paths. Include missing-executable and failed-step tests.

Extend `tests/unit_tests/simtel/test_simtel_file_writer.py` with the three
serializer validation regressions described above. Keep the tests table-based;
do not test geometry or polygon adjacency.

Add `tests/unit_tests/applications/test_plot_trigger_patches.py` to verify CLI
defaults, required model arguments, and delegation to the runner. Remove the
prototype tests for `trigger_patch_validator` and the visualization module.

Add an integration configuration under `tests/integration_tests/config/` only
after a maintained camera model with trigger tables and sim_telarray resources
is identified. It should run the real command, assert the `.ps` and `.simtel.gz`
outputs, and inspect the application logs. It must use generated/static resource
macros and relative output paths.

### 5. Documentation and completion checks

Replace the untracked application page with usage-oriented documentation:
state that the command writes a PostScript display generated through pixled and
sim_telarray, list the retained intermediate artifacts, explain
`--required_pixels` and `--include_presum_slaves`, and include CLI help.

Remove the untracked `trigger_patch_validator` and `visualization` module
references from API documentation. Add an API-reference entry for
`simtools.simtel.trigger_patch_mapping`, update the application page, and keep
the existing alphabetical applications index entry.

For handoff, run the focused serializer, runner, and application unit tests.
Run the new integration test only where sim_telarray and the required resource
bundle are installed. Confirm that the PostScript contains one or more camera
pages and that pixled reports a nonzero number of generated events for the
selected camera model.

## Acceptance criteria

1. The command uses the camera file exported from the selected model tables and
   `pixled --use-trg-all`; no table rows are redrawn or interpreted separately.
2. Each successful run leaves the camera file, LED IACT input, sim_telarray
   output, PostScript display, and logs in `output_path`.
3. A malformed trigger definition fails before execution where the serializer
   can identify it, and external-program failures retain enough logs to debug.
4. The output can be reviewed with a PostScript viewer and demonstrates the
   same mapping sim_telarray actually consumed.
