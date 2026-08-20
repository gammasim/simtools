window.BENCHMARK_DATA = {
  "lastUpdate": 1787233227985,
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
      },
      {
        "commit": {
          "author": {
            "email": "gernot.maier@desy.de",
            "name": "Gernot Maier",
            "username": "GernotMaier"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "13070f9d995d071ff104a5304a0e15899c793e04",
          "message": "Fix failing unit tests in model/test_model_repository.py (#2443)\n\n* Fix failing unit tests\n\n* path fix\n\n* consistent resource benmark package\n\n* docs\n\n* add tests to docs\n\n* simplification\n\n* path settings",
          "timestamp": "2026-08-19T13:41:08+02:00",
          "tree_id": "631d06a4e4032a2ca0277ad56b069b60f2cd6c98",
          "url": "https://github.com/gammasim/simtools/commit/13070f9d995d071ff104a5304a0e15899c793e04"
        },
        "date": 1787140626257,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 51.260404336999954,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260810.271.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 53.1,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260810.271.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 411.79296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260810.271.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 844.270017572,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 995.84,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1066.6484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 6.430054633000012,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 6.9399999999999995,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 519.3515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 5.8927698999999905,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.380000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 429.921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.162558097000016,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 7.55,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 438.4921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.321486950000008,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 4.94,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 554.86328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 35.23896131799998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 28.35,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 410.87109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 6.394608224999985,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 7.01,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 430.5703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 9.55885889199999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 10.16,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 594.9140625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 12.95506862000002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 13.48,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 694.51953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 6.9191646880000235,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 7.6,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 435.81640625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 7.959168298000009,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 8.48,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 463.69140625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.657195916000035,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 6,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 629.4609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.072740044,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.51,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 718.51953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 11.184745959999987,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 10.49,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 552.30078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 18.314087543000028,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 19.13,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 646.74609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 13.09992700500004,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 13.620000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 630.62890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.825813471999993,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 16.29,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 657.89453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 9.073907570000017,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 14.22,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 739.45703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.657819141999994,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.069999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 744.2734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 19.064222653,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 22.6,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1059.96875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.11435171100004,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 5.55,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 606.31640625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 36.56011143699993,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 41.92,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1066.6484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 10.85875365000004,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 12.76,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 863.9140625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 15.128387001999954,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 15.24,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 497.5546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "orel.gueta@desy.de",
            "name": "orelgueta",
            "username": "orelgueta"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "58b92f0a78cd2898157366981d6f0fb96ca7253f",
          "message": "Fix instrument name stripped of OBS- prefix when writing model parameter JSON (#2444)\n\n* Do not strip the OBS from the instrument name\n\n* Long line\n\n* Change log\n\n---------\n\nCo-authored-by: Gernot Maier <gernot.maier@desy.de>",
          "timestamp": "2026-08-19T14:07:34+02:00",
          "tree_id": "0f1ec5877b2c8adce03f2cc8918404260884a562",
          "url": "https://github.com/gammasim/simtools/commit/58b92f0a78cd2898157366981d6f0fb96ca7253f"
        },
        "date": 1787142278756,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 55.369094032999996,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260810.271.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 57.49999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260810.271.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 416.99609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260810.271.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 873.3023057099999,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 1020.1,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1059.078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 6.453938981000007,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 6.9399999999999995,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 517.875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 5.884227836999997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.4,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 426.953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.246700997000005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 7.7,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 438.16015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.178740977999979,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 4.880000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 390.8125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 35.42878565000001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 28.37,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 409.3125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 6.7417065930000035,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 7.2700000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 429.33203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 9.610137973000008,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 10.16,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 592.953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 12.84091721699997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 13.41,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 663.4375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 7.211073529000032,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 7.86,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 428.99609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 8.261483225000006,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 8.760000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 461.37109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.772120240999982,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 6.039999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 629.1796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.342154644999994,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.74,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 715.625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 11.445674888999974,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 10.770000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 546.82421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 18.565997844000037,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 19.2,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 639.44140625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 13.238233449999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 13.700000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 625.21484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.722482205999995,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 16.040000000000003,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 459.6875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 9.318988958000034,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 14.39,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 715.875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.885587187999988,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.279999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 742.953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 19.463571194999986,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 23.240000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1059.078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.454935137999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 5.96,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 601.12109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 37.708111142999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 43.22,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1058.9453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 11.360433973,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 13.43,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 851.47265625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / wall_time_s",
            "value": 5.020620829999984,
            "unit": "wall_time_s",
            "extra": "outcome=skipped | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / cpu_time_s",
            "value": 5.29,
            "unit": "cpu_time_s",
            "extra": "outcome=skipped | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / peak_rss_mib",
            "value": 450.1875,
            "unit": "peak_rss_mib",
            "extra": "outcome=skipped | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 15.801125410000054,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 15.900000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 497.12109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "gernot.maier@desy.de",
            "name": "Gernot Maier",
            "username": "GernotMaier"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ed36cfd01016b33ca5a4fa4d9b90c88bdc9097d7",
          "message": "Follow up fix for benchmark tests (#2447)",
          "timestamp": "2026-08-19T14:35:08+02:00",
          "tree_id": "936b370dd69f70e18769c953ab539047a684fb94",
          "url": "https://github.com/gammasim/simtools/commit/ed36cfd01016b33ca5a4fa4d9b90c88bdc9097d7"
        },
        "date": 1787143901076,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 53.961213177999994,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 55.76,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 414.42578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 814.322975021,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 961.0600000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1048.21875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 6.44623012000001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 6.98,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 518.3515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 5.735449579000004,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.199999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 425.49609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.0682414439999945,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 7.46,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 437.953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.177961334999992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 5,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 393.58984375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 34.615684423000005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 27.85,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 410.34375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 6.276086942000006,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 6.81,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 429.82421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 9.265593723999984,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 9.86,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 594.11328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 12.512966678999987,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 13.059999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 663.640625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 6.664577928999961,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 7.34,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 427.3046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 7.648453558000028,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 8.15,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 459.96875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.540331225000045,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 5.77,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 612.96484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.02826195800003,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.359999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 713.03515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 11.092362793000007,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 10.3,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 536.20703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 18.181663094000044,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 19,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 630.56640625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 12.989043852999998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 13.55,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 613.94921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.726650842999959,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 16.029999999999998,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 418.76953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 8.991473301000042,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 13.769999999999998,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 684.6875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.552831648999984,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 6.950000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 729.12890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 18.837190697999972,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 22.369999999999997,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1048.21875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.080437162000067,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 5.61,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 592.04296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 36.846246782000094,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 42.1,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1047.078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 10.926963475999969,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 12.73,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 846.74609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 14.882710639000038,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 14.83,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 490.6875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "gernot.maier@desy.de",
            "name": "Gernot Maier",
            "username": "GernotMaier"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e7287e3d904b0587e75b6467af4ada69396ee421",
          "message": "Remove apptainer build (#2446)\n\n* Remove apptainer build\n\n* changelog\n\n* Potential fix for pull request finding\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>\n\n* Potential fix for pull request finding\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>\n\n---------\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>",
          "timestamp": "2026-08-19T14:41:01+02:00",
          "tree_id": "1232edc67b28c95bdc46b366c10ee8cfa405d298",
          "url": "https://github.com/gammasim/simtools/commit/e7287e3d904b0587e75b6467af4ada69396ee421"
        },
        "date": 1787144415042,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 53.09209066700001,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 54.74,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 414.98046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 833.76806014,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 982.07,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1056.01171875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 6.504324929999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 7.12,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 518.5,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 5.791106457000012,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.18,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 426.87890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.095770026000025,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 7.6000000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 446.91796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.149737739000045,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 4.78,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 391.765625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 34.72229802499993,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 28.1,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 409.6796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 6.411367879000068,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 6.9399999999999995,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 430.203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 9.392874185999972,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 9.93,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 594.0625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 12.623901939000007,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 13.2,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 663.9296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 6.728239150000036,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 7.480000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 435.203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 7.737469733000012,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 8.299999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 465.12109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.522883726000032,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 5.800000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 619.89453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.05732054200007,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.529999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 721.24609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 11.225762968000026,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 10.65,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 546.5625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 18.230360598999937,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 18.78,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 639.73828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 13.057505736000053,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 13.499999999999998,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 620.921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.592606879000073,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 16.080000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 439.28125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 9.077917921999983,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 13.770000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 685.6953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.646573912000008,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 6.9399999999999995,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 738.84375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 19.051438618999896,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 22.63,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1051.41796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.234045908999974,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 5.659999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 599.58203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 36.92307873900006,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 42.15,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1056.01171875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 11.007803301999957,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 13.04,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 850.140625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 15.31337540900006,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 15.29,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 495.03515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "gernot.maier@desy.de",
            "name": "Gernot Maier",
            "username": "GernotMaier"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5d5737ac3475f25492613b92bda5c27c4adaeb01",
          "message": "Add flags to pytest to printout reasons for skipped and xfailed tests. (#2448)\n\n* Add flags to pytest to printout reasons for skipped and xfailed tests.\n\n* correct PR number\n\n* Potential fix for pull request finding\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>\n\n---------\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>",
          "timestamp": "2026-08-19T15:05:14+02:00",
          "tree_id": "7741892d71b6013efecbc70418015b1c54d0d525",
          "url": "https://github.com/gammasim/simtools/commit/5d5737ac3475f25492613b92bda5c27c4adaeb01"
        },
        "date": 1787145677603,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 52.24374975800001,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260810.271.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 54.209999999999994,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260810.271.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 415.16015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260810.271.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 838.1099574950001,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 986.19,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1056.31640625,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 6.80668388700002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 7.26,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 518.359375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 6.1330837570000085,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.529999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 427.5625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.409790391000001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 7.8,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 437.9921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.3799060030000305,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 5.079999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 396.85546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 35.38770667,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 28.65,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 410.36328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 6.454992435000008,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 6.97,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 430.06640625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 9.345684056999971,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 9.809999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 592.76953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 12.543176288999973,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 13.120000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 663.515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 6.708440542999938,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 7.359999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 426.23828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 7.730500283999959,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 8.249999999999998,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 465.3515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.676450919999979,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 5.94,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 621.375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.027370078999979,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.32,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 719.0546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 11.16374310599997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 10.540000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 546.84375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 18.31790357700004,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 18.9,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 638.95703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 12.98301668800002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 13.5,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 624.515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.63250494099998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 16.12,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 436.66796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 9.13864262800007,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 14.280000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 703.03515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.851244414000007,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.3,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 738.79296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 19.07336655200004,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 22.599999999999998,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1056.31640625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.229517933000011,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 5.67,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 599.41796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 36.95557880000001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 42.16,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1056.0078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 11.065696667999987,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 13.07,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 859.55078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 15.111888461000035,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 15.2,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 493.77734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          }
        ]
      },
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
          "id": "5d5737ac3475f25492613b92bda5c27c4adaeb01",
          "message": "Add flags to pytest to printout reasons for skipped and xfailed tests. (#2448)\n\n* Add flags to pytest to printout reasons for skipped and xfailed tests.\n\n* correct PR number\n\n* Potential fix for pull request finding\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>\n\n---------\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>",
          "timestamp": "2026-08-19T13:05:14Z",
          "url": "https://github.com/gammasim/simtools/commit/5d5737ac3475f25492613b92bda5c27c4adaeb01"
        },
        "date": 1787191851974,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 51.33818925100002,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 53.24,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 415.0078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 550.6632185689999,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 694.46,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1051.7421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 22.859909751999993,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 18.580000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 409.93359375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 6.08874289100001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 6.71,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 593.12890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 8.315253909000006,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 8.81,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 663.8125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 5.119546235999962,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 5.55,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 460.78515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 8.019794339999976,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 7.28,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 539.46484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 12.706434994000006,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 13.29,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 634.55859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 8.916399259999992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 9.469999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 618.44921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 9.97887286599996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 10.43,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 429.62109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 5.954976784999985,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 8.74,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 742.1875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 12.17632379500003,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 14.41,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1050.96484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 23.934859404999997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 27.6,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1051.7421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 7.5835801090000245,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 9.06,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 848.421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 9.678801436000015,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 9.899999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 493.75,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "gernot.maier@desy.de",
            "name": "Gernot Maier",
            "username": "GernotMaier"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0ecb3cf0977848bb262f42027b5cb82af99e7525",
          "message": "Correct treatment of `ct` vs `count` for simulation models. (#2451)",
          "timestamp": "2026-08-20T09:54:53+02:00",
          "tree_id": "5c63c3b33e7007afcf9e14d1e3451b2778fdc0f3",
          "url": "https://github.com/gammasim/simtools/commit/0ecb3cf0977848bb262f42027b5cb82af99e7525"
        },
        "date": 1787213485750,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 40.621669613999984,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260810.271.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 40.95,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260810.271.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 416.046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260810.271.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 874.6657891169999,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 1022.11,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1053.9921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 6.749953529999999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 7.26,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 518.28515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 6.0331747840000105,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.43,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 427.58984375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.499139637000013,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 7.99,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 438.71484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.480640894999993,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 5.1899999999999995,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 396.40234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 35.43151741999998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 28.53,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 409.9765625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 6.60029564599995,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 7.03,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 430.1328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 9.661922564999998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 10.129999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 594.328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 12.741908453000008,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 13.32,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 664.3984375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 6.821531471000014,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 7.39,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 430.51171875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 7.760510317000012,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 8.280000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 462.73046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.619635899999992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 5.69,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 625.30859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.115443013999993,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.360000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 716.57421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 11.329607923000026,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 10.620000000000003,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 530.46484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 18.234539774999973,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 18.84,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 638.3203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 13.186592934999965,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 13.870000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 624.6015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.651190424000106,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 15.93,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 452.23046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 9.104266232999976,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 13.78,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 692.015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.88558220699997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.340000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 733.52734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 19.431435609999994,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 23.110000000000003,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1051.22265625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.292590931000063,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 5.680000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 596.10546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 37.3211991820001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 42.699999999999996,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1053.9921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 11.25165585800005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 13.27,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 849.15234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 16.232526138000026,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 16.39,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 493.5234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "gernot.maier@desy.de",
            "name": "Gernot Maier",
            "username": "GernotMaier"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "216eb9c9759df6134d1b2e76a5b3c6d63569b3f2",
          "message": "Keep consistent UUIDs for temporary files when writing reduced event lists. (#2452)\n\n* Keep consistent UUIDs for temporary files when writing reduced event lists\n\n* Potential fix for pull request finding\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>\n\n---------\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>",
          "timestamp": "2026-08-20T10:14:03+02:00",
          "tree_id": "948354f9b6d14d5f8eda6edf8d7bd4d365f20c99",
          "url": "https://github.com/gammasim/simtools/commit/216eb9c9759df6134d1b2e76a5b3c6d63569b3f2"
        },
        "date": 1787214434251,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 55.77106764700005,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 57.66,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 415.15234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 658.495764552,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 800.74,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1045.1328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 5.216113314000012,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 5.640000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 517.44921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 5.688787362999989,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 6.1,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 437.109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 28.009788654999994,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 22.05,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 408.875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 7.417520388000014,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 8.04,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 591.70703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 10.06171120600004,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 10.650000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 662.20703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 5.475346269999989,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 6.200000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 428.54296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 6.072202636999975,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 6.659999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 456.89453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 5.5907798439999965,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 5.97,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 707.07421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 8.770532129000003,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 8.29,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 512.44140625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 14.174235828999997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 14.98,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 632.07421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 10.169865254000001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 10.690000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 615.33203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 12.208835015000034,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 12.47,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 388.75390625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 7.0668772600000125,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 10.32,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 690.93359375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 5.204353545000004,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 5.7,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 728.390625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 14.82020229899996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 17.46,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1045.1328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 28.48925903999998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 32.589999999999996,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1045,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 8.589989134000007,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 10.24,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 850.26953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 11.931599235999897,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 12.059999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 489.94140625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "gernot.maier@desy.de",
            "name": "Gernot Maier",
            "username": "GernotMaier"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8dc9226bd9c36925de2d09fe87356d748457bbbe",
          "message": "Relax fine-tuned test limits in integration tests (#2456)\n\n* Relax fine-tuned test limits in integration tests\n\n* fixed limits",
          "timestamp": "2026-08-20T15:23:42+02:00",
          "tree_id": "495751d1499d1e4db66fe30011aed06237dcb3db",
          "url": "https://github.com/gammasim/simtools/commit/8dc9226bd9c36925de2d09fe87356d748457bbbe"
        },
        "date": 1787233213529,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 55.471559588999995,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 57.38999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 415.26171875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 862.2391391230001,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 1020.22,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1055.16796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 9.638007312000013,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 10.2,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 558.5390625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 6.20613048499996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.630000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 426.84765625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.62129905300003,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 8.01,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 439.11328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.532020870999986,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 5.379999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 396.81640625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 33.38164022199999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 26.81,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 410.00390625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 6.995018156000015,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 7.48,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 431.02734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 10.546412093000015,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 11.07,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 594.7734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 14.677866937000033,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 15.2,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 698.53125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 7.964989275999983,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 8.799999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 429.25,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 8.27408334100005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 8.91,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 465.0703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.974919077999971,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 6.259999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 613.91796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.149858421999966,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.589999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 716.43359375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 10.650298907999968,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 10.19,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 531.984375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 17.061204176999922,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 17.849999999999998,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 640.921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 12.21982682700002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 12.57,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 626.80859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.295222856999999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 15.74,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 454.78515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 8.845736246000001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 13.93,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 703.5703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.96696003999989,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.330000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 735.86328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 19.148157603999948,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 22.89,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1053.703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.41975053200008,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 6.0200000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 598.68359375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_sim_telarray_only / wall_time_s",
            "value": 5.035548527999936,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_sim_telarray_only / cpu_time_s",
            "value": 5.5699999999999985,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_sim_telarray_only / peak_rss_mib",
            "value": 495.5078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 36.03063723299999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 40.94,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1055.16796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 11.189553606999993,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 13.250000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 855.0703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 18.59218935399997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 18.630000000000006,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 497.35546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          }
        ]
      }
    ]
  }
}