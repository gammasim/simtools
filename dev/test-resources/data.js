window.BENCHMARK_DATA = {
  "lastUpdate": 1787143914541,
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
      }
    ]
  }
}