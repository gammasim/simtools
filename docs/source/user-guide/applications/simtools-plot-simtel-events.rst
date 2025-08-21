simtools-plot-simtel-events
=================================

Plot sim_telarray event products using simtools visualization utilities. This app
can be run after simulations as
``simtools-simulate-flasher`` or ``simtools-simulate-illuminator``.

Usage
-----

.. code-block:: console

    simtools-plot-simtel-events \
      --simtel_files <file1.simtel.gz> [<file2.simtel.gz> ...] \
      --plots <plot1> [<plot2> ...] \
      [--tel_id 1] [--output_file my_plots] [--save_pngs]

Arguments
---------

- ``--simtel_files`` (required): one or more ``.simtel.gz`` files produced by sim_telarray.
- ``--plots`` (optional): select which plots to generate. Choices:
  ``event_image``, ``time_traces``, ``waveform_pcolormesh``, ``step_traces``,
  ``integrated_signal_image``, ``integrated_pedestal_image``, ``peak_timing``, or ``all``.
  Default is ``event_image``.
- ``--tel_id`` (optional): telescope id to visualize (default: first available).
- ``--n_pixels`` (optional): for time_traces (default: 3).
- ``--pixel_step`` (optional): for step_traces and waveform_pcolormesh (default: 100).
- ``--max_pixels`` (optional): cap number of pixels for step_traces.
- ``--vmax`` (optional): color scale upper limit for waveform_pcolormesh.
- ``--half_width`` (optional): window half-width for integrated_*_image (default: 8).
- ``--gap`` (optional): pedestal gap for integrated_pedestal_image (default: 16).
- ``--sum_threshold`` ``--peak_width`` ``--examples`` ``--timing_bins`` (optional): parameters for ``peak_timing``.
- ``--distance`` (optional): annotation for event_image.
- ``--output_file`` (optional): base name used to create multi-page PDFs under simtools output dir.
- ``--save_pngs`` (flag): also save individual PNGs per plot.
- ``--dpi`` (optional): PNG DPI (default 300).

Examples
--------

1) Camera image and time traces for a single file, save a PDF:

.. code-block:: console

    simtools-plot-simtel-events \
      --simtel_files simtools-output/simulate_illuminator/xyzls.simtel.gz \
      --plots event_image time_traces \
      --tel_id 1 \
      --output_file illuminator_inspect

2) All plots for multiple files, PNGs and PDFs:

.. code-block:: console

    simtools-plot-simtel-events \
      --simtel_files f1.simtel.gz f2.simtel.gz \
      --plots all \
      --save_pngs --dpi 200

Notes
-----

- Use this app to create plots from the produced ``.simtel.gz`` files.
- Multi-page PDFs and optional PNGs are written under the standard simtools output
  directory, following the usual IOHandler conventions.
