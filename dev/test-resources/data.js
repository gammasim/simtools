window.BENCHMARK_DATA = {
  "lastUpdate": 1787105923788,
  "repoUrl": "https://github.com/gammasim/simtools",
  "entries": {
    "simtools CI test benchmarks": [
      {
        "commit": {
          "author": {
            "name": "Gernot Maier",
            "username": "GernotMaier",
            "email": "gernot.maier@desy.de"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "d1c11bc6d8b601b0cd2261a8c7f68bbd3a61363e",
          "message": "Integration test benchmark (#2435)\n\n* Testing benchmarking\n\n* test benchmark\n\n* simplification\n\n* permissions, conflicts\n\n* remove workflow duplication\n\n* missing action\n\n* changelog\n\n* unit tests\n\n* reviewers comments",
          "timestamp": "2026-08-18T12:18:17Z",
          "url": "https://github.com/gammasim/simtools/commit/d1c11bc6d8b601b0cd2261a8c7f68bbd3a61363e"
        },
        "date": 1787105908042,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-suite / wall_time_s",
            "value": 54.152140633,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260810.271.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-suite / cpu_time_s",
            "value": 56.120000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260810.271.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-suite / peak_rss_mib",
            "value": 413.08984375,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260810.271.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-suite / 7.0.0 / wall_time_s",
            "value": 876.9740039000001,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-suite / 7.0.0 / cpu_time_s",
            "value": 1038.43,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-suite / 7.0.0 / peak_rss_mib",
            "value": 1061.49609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 6.739451938000016,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 7.24,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 518.75,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 6.0432501049999985,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.48,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 427.90234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.421528633000008,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 7.83,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 448.58203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.2882173069999965,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 4.99,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 555.54296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 33.47240118299999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 26.61,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 410.484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 7.055010600999992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 7.62,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 430.515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-array-layout_from_list_north / wall_time_s",
            "value": 5.042709576999982,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-array-layout_from_list_north / cpu_time_s",
            "value": 5.689999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-array-layout_from_list_north / peak_rss_mib",
            "value": 423.37109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 10.942730369000003,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 11.51,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 592.640625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 14.494691278999994,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 15.08,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 694.28125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-tabular-data-for-model-parameter_atmospheric_profile_all / wall_time_s",
            "value": 5.102025107999964,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-tabular-data-for-model-parameter_atmospheric_profile_all / cpu_time_s",
            "value": 5.6499999999999995,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-tabular-data-for-model-parameter_atmospheric_profile_all / peak_rss_mib",
            "value": 400.33984375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 8.369937350999976,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 9.2,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 428.52734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 8.851472858000022,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 9.48,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 463.41015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 6.032413971999972,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 6.44,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 628.55078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.377827719999971,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.8500000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 717.06640625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 10.818579983999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 10.22,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 550.88671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 17.202951869999993,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 17.84,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 645.72265625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 12.293818541000007,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 12.84,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 632.265625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.24196465,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 15.61,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 660.62109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 8.88022636300002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 13.6,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 733.85546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.973893773999976,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.24,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 742.51171875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 18.175001855999994,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 21.639999999999997,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1056.52734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.544527010000024,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 6.17,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 603.99609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 36.53839877100006,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 41.68,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1061.49609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 10.635173410999982,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 12.469999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 846.33203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_MSTN / wall_time_s",
            "value": 5.031201618999944,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_MSTN / cpu_time_s",
            "value": 5.319999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_MSTN / peak_rss_mib",
            "value": 448.7265625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / wall_time_s",
            "value": 5.110459526999989,
            "unit": "wall_time_s",
            "extra": "outcome=skipped | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / cpu_time_s",
            "value": 5.48,
            "unit": "cpu_time_s",
            "extra": "outcome=skipped | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / peak_rss_mib",
            "value": 454.1953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=skipped | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 18.71576784800004,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 18.86,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 495.75,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          }
        ]
      }
    ]
  }
}