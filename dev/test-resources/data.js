window.BENCHMARK_DATA = {
  "lastUpdate": 1788156863322,
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
          "id": "b0a7b5f87c69fb3d236688bc4c6240ad7beb12f9",
          "message": "Ensure consistent version tagging for simtools-prod and simtools-dev images (#2459)\n\n* Ensure consistent version tagging for simtools-prod and simtools-dev images.\n\n* correct PR number [skip ci]",
          "timestamp": "2026-08-20T14:54:12Z",
          "url": "https://github.com/gammasim/simtools/commit/b0a7b5f87c69fb3d236688bc4c6240ad7beb12f9"
        },
        "date": 1787278901960,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 53.40550185500001,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 55.6,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 415.3828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 698.25848014,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 866.59,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1059.79296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 7.143744486999992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 7.78,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 567.9375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 5.75094640399999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 6.34,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 439.14453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 29.184273933000014,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 23.52,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 412.0234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 5.280264402,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 5.92,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 435.04296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 7.566045122000048,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 8.209999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 595.09375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 10.994127521000053,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 11.7,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 695.2734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 5.567224354000018,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 6.46,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 433.33203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 6.275807993000001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 6.8100000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 462.91796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 5.823465222999971,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 6.249999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 716.53515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 9.734307713000021,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 9.5,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 544.80078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 15.886670555000023,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 16.6,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 639.05859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 11.359723298000006,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 11.73,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 624.26171875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 12.939024315999973,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 13.43,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 456.49609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 7.478269038999997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 11.620000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 768.92578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 5.592433282000002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 6.039999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 735.12109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 15.273880528999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 18.1,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1054.55859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 31.344551474000014,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 35.88,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1059.79296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 8.949562447999995,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 10.51,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 857.29296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 12.374097161000009,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 12.57,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 499.2421875,
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
          "id": "a4fdb316876f9f1a0babda81edb426f2ba395eff",
          "message": "Merge pull request #2461 from gammasim/small_additions_to_psf_plot\n\nAllow to plot PSF images also in degrees and add the name of the telescope to the annotation",
          "timestamp": "2026-08-21T11:34:17+02:00",
          "tree_id": "d15c54a0066d198c569e4a75e1adc477f0089ad4",
          "url": "https://github.com/gammasim/simtools/commit/a4fdb316876f9f1a0babda81edb426f2ba395eff"
        },
        "date": 1787305675165,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 55.48911109299999,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 57.19,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 415.9453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 676.803568883,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 822.86,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1054.80078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 6.983078720999998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 7.470000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 568.65625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 5.7318116770000245,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 6.23,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 440.328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 27.626290839999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 22.23,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 414.32421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 5.2220091210000135,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 5.82,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 434.421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 7.44942309999999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 8.030000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 594.91796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 10.444051305000016,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 11.02,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 695.17578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 5.411317159999953,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 6.089999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 422.21484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 6.1083449209999685,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 6.740000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 462.16015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 5.562584788000038,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 5.840000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 716.921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 8.79180098400002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 7.960000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 500.28125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 14.20933155900002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 14.669999999999998,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 633.5234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 10.154656418999991,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 10.55,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 621.6015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 12.200490007999974,
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
            "value": 398.515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 7.102069531000041,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 10.19,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 706.23828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 5.27159533300005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 5.5600000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 734.07421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 14.884309143999985,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 17.62,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1051.60546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 28.982429939999975,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 33.2,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1054.80078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 8.735200127999974,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 10.47,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 858.01953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 12.096274439999888,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 12.149999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 499.67578125,
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
          "id": "1cafba177b76fda441b10f12abb4d3fa4c4910eb",
          "message": "Update and improve release documentation (#2460)\n\n* Update and improve release documentation\n\n* Potential fix for pull request finding\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>\n\n* correct PR number\n\n---------\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>",
          "timestamp": "2026-08-21T13:04:25+02:00",
          "tree_id": "879cfa23d213ae123347de43c234cb7df0b476a6",
          "url": "https://github.com/gammasim/simtools/commit/1cafba177b76fda441b10f12abb4d3fa4c4910eb"
        },
        "date": 1787311246222,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 57.378777009000004,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 59.83,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 416.23828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 859.5117970370001,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 1018.93,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1051.421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 9.441687283999997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 9.940000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 569.20703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 6.109803921000008,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.699999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 431.8671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.594284929999986,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 8,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 441.1015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.478874895999979,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 5.299999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 401.87109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 33.441106647,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 26.819999999999997,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 414.7890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 7.0985512889999995,
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
            "value": 435.76953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 10.470362111000043,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 11,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 596.60546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 14.708114582000007,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 15.25,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 696.171875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 8.119802660999994,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 8.959999999999997,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 437.5546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 8.516910478,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 9.139999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 453.1484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 6.012487444999977,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 6.17,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 618.42578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.222916924999993,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.550000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 713.7578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 10.715134147000015,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 10.02,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 518.890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 17.19082666700001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 17.89,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 636.390625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 12.214961700000003,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 12.93,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 631.9375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.417800460999956,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 15.59,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 389.98828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 8.96335620299999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 14.070000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 698.88671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.96594239999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.38,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 728.14453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 18.92520495399998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 22.5,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1051.421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.5624212010000065,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 6.03,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 596.99609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 36.02515301800008,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 41.120000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1050.94921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 11.183102989999952,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 13.219999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 854.66015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / wall_time_s",
            "value": 5.040193485999907,
            "unit": "wall_time_s",
            "extra": "outcome=skipped | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / cpu_time_s",
            "value": 5.41,
            "unit": "cpu_time_s",
            "extra": "outcome=skipped | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / peak_rss_mib",
            "value": 449.6953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=skipped | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 18.50334344800001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 18.65,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 495.46875,
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
          "id": "cac62b39a87b3ea7a8068ec29de633f0fbef51ed",
          "message": "Merge pull request #2462 from gammasim/view_cone_metadata_round\n\nRound the view cone values in the metadata to two decimal points",
          "timestamp": "2026-08-21T14:00:36+02:00",
          "tree_id": "6538c125d75957d51796f2616c8752483a58048d",
          "url": "https://github.com/gammasim/simtools/commit/cac62b39a87b3ea7a8068ec29de633f0fbef51ed"
        },
        "date": 1787314508656,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 54.785013436,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 56.790000000000006,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 416.140625,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 714.641658909,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 882.13,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1060.6015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 7.502376972999997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 8.19,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 568.30078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 6.008194216000021,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 6.459999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 435.9609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 30.27042491900002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 24.5,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 413.26953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 5.372822080999981,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 5.75,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 432.95703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 7.573406749000014,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 8.149999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 594.32421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 11.352168576999986,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 12,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 693.3125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 5.569115192000027,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 6.5,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 434.14453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 6.3511681840000165,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 7.009999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 460.0234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 6.0500395660000095,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 6.6,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 712.09375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 10.027774145000024,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 9.530000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 535.96875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 16.616102807000004,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 17.06,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 638.18359375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 11.763890613000001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 12.370000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 624.6953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 13.49743321599999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 14.08,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 439.95703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 7.905604922000009,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 12.25,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 717.078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 5.732054918000017,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 6.170000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 738.28515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 15.805032050999955,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 18.75,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1055.87109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 32.65390378000001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 37.199999999999996,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1060.6015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 9.160514010000043,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 11.09,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 856.88671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 13.019512573999918,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 13.17,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 498.703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "orelgueta",
            "username": "orelgueta",
            "email": "orel.gueta@desy.de"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "cac62b39a87b3ea7a8068ec29de633f0fbef51ed",
          "message": "Merge pull request #2462 from gammasim/view_cone_metadata_round\n\nRound the view cone values in the metadata to two decimal points",
          "timestamp": "2026-08-21T12:00:36Z",
          "url": "https://github.com/gammasim/simtools/commit/cac62b39a87b3ea7a8068ec29de633f0fbef51ed"
        },
        "date": 1787364831061,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 56.32090774100001,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 58.48999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 416.0234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 848.4269994709999,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 994.84,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1050.3046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 8.967291037999985,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 9.48,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 569.65625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 6.043989957999997,
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
            "value": 426.546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.34388177400001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 7.800000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 438.13671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.236278189000018,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 4.970000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 399.5703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 35.217730028999995,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 28.4,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 413.328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 6.6356938600000035,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 7.24,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 434.8671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 9.454815122000014,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 9.95,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 595.3671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 13.419071299999985,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 13.969999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 696.39453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 6.827332013999978,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 7.66,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 430.71484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 7.810817390000011,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 8.2,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 454.46484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.564332589999992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 5.8100000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 631.24609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.062343434000013,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.44,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 717.3046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 11.215649453999958,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 10.440000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 542.12109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 18.24833462800001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 19.06,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 632.78515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 13.18406238600005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 13.64,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 625.375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.605207318999987,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 16.05,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 449.296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 9.075061011999992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 13.42,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 694.2265625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.678302217999999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.03,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 731.4453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 19.244264840000028,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 22.87,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1047.12890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.166051985999957,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 5.66,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 592.828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 37.036705547,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 42.17,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1050.3046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 11.10184219200005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 13.15,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 854.22265625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 15.506250623000028,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 15.46,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 494.859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "orelgueta",
            "username": "orelgueta",
            "email": "orel.gueta@desy.de"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "cac62b39a87b3ea7a8068ec29de633f0fbef51ed",
          "message": "Merge pull request #2462 from gammasim/view_cone_metadata_round\n\nRound the view cone values in the metadata to two decimal points",
          "timestamp": "2026-08-21T12:00:36Z",
          "url": "https://github.com/gammasim/simtools/commit/cac62b39a87b3ea7a8068ec29de633f0fbef51ed"
        },
        "date": 1787451894346,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 57.28363299299997,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 59.480000000000004,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 416.3046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 784.929282016,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 956.47,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1057.08203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 7.934807074000005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 8.600000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 567.94921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 5.316893161999985,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 5.8,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 426.0703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 6.381122385999987,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 6.86,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 436.375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 31.662208429000003,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 25.7,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 413.65625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 6.013524797000002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 6.529999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 434.7578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 8.412260500000002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 8.969999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 593.03515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 12.317402852999976,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 13,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 692.84375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 6.034121775000017,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 6.96,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 421.44140625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 6.866247380999994,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 7.53,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 460.8984375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.127879751000023,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 5.409999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 620.2421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 6.3964006119999794,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 6.72,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 711.69140625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 10.540395971999999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 10.180000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 539.2421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 17.31564516000003,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 18.1,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 635.8828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 12.323731597999995,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 12.72,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 621.109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 14.052541605999977,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 14.56,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 441.3046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 8.269160387999989,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 12.67,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 730.87890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.060416610000004,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 6.5,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 734.578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 16.66280051199999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 19.76,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1056.96875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 34.051499511999964,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 38.690000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1057.08203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 9.877830730000028,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 11.819999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 852.16796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 13.677676202999919,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 13.899999999999997,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 500.453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "orelgueta",
            "username": "orelgueta",
            "email": "orel.gueta@desy.de"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "cac62b39a87b3ea7a8068ec29de633f0fbef51ed",
          "message": "Merge pull request #2462 from gammasim/view_cone_metadata_round\n\nRound the view cone values in the metadata to two decimal points",
          "timestamp": "2026-08-21T12:00:36Z",
          "url": "https://github.com/gammasim/simtools/commit/cac62b39a87b3ea7a8068ec29de633f0fbef51ed"
        },
        "date": 1787538278058,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 55.80428326399988,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 58.040000000000006,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 416.5078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 876.6865615019999,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 1033.96,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1056.62890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 9.304885354000007,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 9.96,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 569.17578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 6.2492035330000135,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.640000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 429.41015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.707725443000015,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 8.120000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 436.515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.478194629000001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 5.129999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 400.08203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 33.63566390700001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 27.05,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 412.9453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 7.082894991999979,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 7.7,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 434.0859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-array-layout_from_list_north / wall_time_s",
            "value": 5.119437771000037,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-array-layout_from_list_north / cpu_time_s",
            "value": 5.59,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-array-layout_from_list_north / peak_rss_mib",
            "value": 424.0859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 11.304555569999991,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 11.76,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 594.62109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 15.175267229999974,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 15.73,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 694.35546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 8.353598002000012,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 9.18,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 426.32421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 8.600504524999963,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 9.12,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 459.78515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.960623856999973,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 6.069999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 594.97265625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.292501936999997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.77,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 717.3203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 10.773813651000012,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 10.080000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 525.859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 17.165446351000014,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 18.060000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 642.2578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 12.423889078000002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 12.89,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 631.43359375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.19853759199998,
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
            "value": 397.88671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 8.968445661999965,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 13.430000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 689.70703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 7.0617912090000345,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.6,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 737.328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 19.236739301999933,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 23.03,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1054.6875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.57770798599995,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 6.069999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 600.62890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_sim_telarray_only / wall_time_s",
            "value": 5.263189857000043,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_sim_telarray_only / cpu_time_s",
            "value": 5.640000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_sim_telarray_only / peak_rss_mib",
            "value": 500.23828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 36.51380900899994,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 41.51,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1056.62890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 11.348881731000006,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 13.219999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 855.4921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / wall_time_s",
            "value": 5.052955077999968,
            "unit": "wall_time_s",
            "extra": "outcome=skipped | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / cpu_time_s",
            "value": 5.51,
            "unit": "cpu_time_s",
            "extra": "outcome=skipped | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / peak_rss_mib",
            "value": 456.90234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=skipped | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 18.503070412999932,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 18.739999999999995,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 497.03515625,
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
          "id": "c75356be774e14b1db05241595c3151156b00d36",
          "message": "Fill trigger histograms from many files in a directory (#2464)\n\n* Fill trigger histograms from many files in a directory\n\n* documentation\n\n* changelog\n\n* mount path\n\n* remove event data file from histograms\n\n* fix python path in htcondor backend\n\n* input file lists\n\n* simplified input file lists\n\n* integration test\n\n* output path\n\n* not needed",
          "timestamp": "2026-08-24T12:18:57+02:00",
          "tree_id": "2dbc814930503e3fc9deda7624d6c4611e15e01f",
          "url": "https://github.com/gammasim/simtools/commit/c75356be774e14b1db05241595c3151156b00d36"
        },
        "date": 1787567773462,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 57.069327593,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 59.32,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 416.12890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 841.4313621470001,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 1003.99,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1052.67578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 9.172435038000003,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 9.760000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 571.94921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 6.1475343379999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.65,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 429.40234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.4398132989999795,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 7.94,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 439.703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.267816096000047,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 5.0200000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 397.63671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 32.79296103699994,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 26.240000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 413.5234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 6.82631400899993,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 7.4,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 438.10546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 10.187963103000016,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 10.819999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 597.58984375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 14.22273923199998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 14.93,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 698.98046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 7.832471183000052,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 8.620000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 430.31640625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 8.075788403000047,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 8.7,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 463.0703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.944775843999992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 6.15,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 623.12109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.110023213999966,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.5,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 718.85546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 10.63144646100011,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 9.91,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 546.59375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 17.08862259700004,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 17.57,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 638.45703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 12.161496940000006,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 12.82,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 635.8828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.032053640999948,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 15.32,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 394.921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 8.839487319,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 13.790000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 709.03515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.7171943439999495,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.309999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 739.02734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 18.659378407999952,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 22.39,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1052.67578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.357756341000027,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 5.780000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 597.21875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 35.68619484499993,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 40.76,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1051.9609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 11.056561654999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 12.94,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 853.7265625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 18.149841039000194,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 18.240000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 497.13671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-write-trigger-histograms_write_trigger_histograms_from_directory / wall_time_s",
            "value": 5.073695033999911,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-write-trigger-histograms_write_trigger_histograms_from_directory / cpu_time_s",
            "value": 5.540000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-write-trigger-histograms_write_trigger_histograms_from_directory / peak_rss_mib",
            "value": 522.53125,
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
          "id": "f8fc731047d19a4fd9190c4c5012a8ca09ae4c40",
          "message": "Add machine-readable statistics to production comparisons (#2449)\n\n* test statistics for production comparison\n\n* schema\n\n* bin edges\n\n* annoying  matplotlib\n\n* missing doc\n\n* improve CL arguments; fix plotting of pdf or pngs\n\n* integration tests\n\n* core distance bin width\n\n# Conflicts:\n#\tsrc/simtools/visualization/plot_event_level_production_comparison.py\n\n* Fill trigger histograms from many files in a directory (#2464)\n\n* Fill trigger histograms from many files in a directory\n\n* documentation\n\n* changelog\n\n* mount path\n\n* remove event data file from histograms\n\n* fix python path in htcondor backend\n\n* input file lists\n\n* simplified input file lists\n\n* integration test\n\n* output path\n\n* not needed\n\n* Tighten comparison statistics schema\n\n* Add PR 2449 changelog\n\n* simplify comment\n\n* Restore comparison level option\n\n* Compare array layouts separately\n\n* Document comparison plot return value",
          "timestamp": "2026-08-24T13:45:59+02:00",
          "tree_id": "82a4625b1b4c975a0cb81d52f70ffa7d61bc73b9",
          "url": "https://github.com/gammasim/simtools/commit/f8fc731047d19a4fd9190c4c5012a8ca09ae4c40"
        },
        "date": 1787573029378,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 43.589576964,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 42.43000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 415.0546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 850.989985497,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 1012.63,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1050.9609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 9.550230190000093,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 10.07,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 568.20703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 6.182205137999972,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.77,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 430.90625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.473853725000026,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 7.989999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 439.80078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.262287912000033,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 5.140000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 399.8046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 33.10128193999992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 26.57,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 412.5859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 6.928289502999974,
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
            "value": 436.2109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 10.427565181999967,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 11.04,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 596.9453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 14.500098712000067,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 15.070000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 696.0234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 7.85543157099994,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 8.62,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 430.11328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 8.26704413099992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 8.930000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 463.59375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.859995916000003,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 6.24,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 628.8125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.129621019000069,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.64,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 725.16796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 10.65219209299994,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 10.209999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 540.36328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 17.00311112999998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 17.470000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 630.85546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 12.166266891000078,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 12.97,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 631.03515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.056370648999973,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 15.41,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 388.6328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 8.841415574999928,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 13.51,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 686.3828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.784302084000046,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.25,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 728.84765625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 18.698187716999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 22.350000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1050.9609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.474980804999973,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 6.159999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 597.546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 35.85116708400005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 40.82,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1050.13671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 11.033470095999974,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 12.9,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 849.93359375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 18.293965841000045,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 18.49,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 495.21875,
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
          "id": "919ea5759ed6bd85729360d39d234218e17004a6",
          "message": "Align error message for reading simulation model from path and from DB (#2467)\n\n* Align error message for reading simulation model from path and from DB\n\n* correct telescope naming\n\n* changelog\n\n* consistent arguments",
          "timestamp": "2026-08-24T13:56:46+02:00",
          "tree_id": "609203d784a78a724e2f5273ba8139fad29991fc",
          "url": "https://github.com/gammasim/simtools/commit/919ea5759ed6bd85729360d39d234218e17004a6"
        },
        "date": 1787573971636,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 56.00411585100005,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 58.10999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 416.3125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 863.28425989,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 1024.05,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1052.2578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 9.663016056000004,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 10,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 569.23828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 6.212226068999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.719999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 429.328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.597245942000001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 8.110000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 438.4609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.349570905999997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 5.07,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 396.33203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 33.418529702,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 26.87,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 411.5546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 6.911256454000011,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 7.510000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 435.0078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 10.679870620000031,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 11.159999999999998,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 595.19140625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 14.452454738000029,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 14.99,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 695.5625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 7.842618818000005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 8.55,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 425.609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 8.172911269999986,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 8.780000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 462.58203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.9427518709999845,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 6.31,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 618.59375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.142620637999983,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.510000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 717.12890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 10.655484521999995,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 9.87,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 528.6171875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 17.136568436000005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 17.92,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 637.3359375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 12.189750339,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 12.7,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 618.578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.062801288999992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 15.280000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 391.8671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 8.819688348999989,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 12.81,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 667.48046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.795749290999993,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.18,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 733.39453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 18.78354507000006,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 22.38,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1050.52734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.466741420999938,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 6.130000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 596.98046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 36.05231143800006,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 41.18,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1052.2578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 11.04175941599999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 13.06,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 852.9296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 18.085148512000046,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 18.18,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 498.703125,
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
          "id": "0fe704ec183eb1ee61f5d52e5b06fde94ad7d2bb",
          "message": "Improve documentation for production_derive_corsika_limits (#2468)\n\n* Improve documentation for production_derive_corsika_limits\n\n* changelog\n\n* t pus\n\n* changelog name [skip ci]",
          "timestamp": "2026-08-24T13:47:06Z",
          "url": "https://github.com/gammasim/simtools/commit/0fe704ec183eb1ee61f5d52e5b06fde94ad7d2bb"
        },
        "date": 1787624723889,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 59.837993057999995,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 61.809999999999995,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 416.78515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 853.433217761,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 1012.66,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1049.9296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 9.492875185999992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 10.03,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 566.72265625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 6.271947410999985,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.87,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 427.38671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.552895231000008,
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
            "value": 439.671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.270706220999983,
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
            "value": 397.88671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 33.282983427000005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 26.73,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 410.6171875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 6.955194210000002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 7.4799999999999995,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 434.125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 10.809939814000018,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 11.46,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 594.03515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 14.543026008000027,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 15.13,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 696.5703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 8.069312867999997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 8.829999999999998,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 422.69921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 8.348533906,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 8.92,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 461.0859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.87655945299997,
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
            "value": 607.55859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.082207270000026,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.579999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 717.27734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 10.582817562999992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 9.89,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 541.02734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 17.02399829000001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 17.669999999999998,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 633.7890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 12.223288136000008,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 13.01,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 630.71484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.04617140900001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 15.290000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 391.6875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 8.777706077000005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 13.48,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 682.83203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.784591291999959,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.3500000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 733.0078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 18.038710639999977,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 21.54,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1044.0859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.544904126000006,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 6.0699999999999985,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 592.66796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_sim_telarray_only / wall_time_s",
            "value": 5.008472164000068,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_sim_telarray_only / cpu_time_s",
            "value": 5.340000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_sim_telarray_only / peak_rss_mib",
            "value": 489.68359375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 35.998140734,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 40.89,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1049.9296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 10.475643815000012,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 12.19,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 847.65234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 18.60842857199998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 18.64,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 493.125,
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
          "id": "0fe704ec183eb1ee61f5d52e5b06fde94ad7d2bb",
          "message": "Improve documentation for production_derive_corsika_limits (#2468)\n\n* Improve documentation for production_derive_corsika_limits\n\n* changelog\n\n* t pus\n\n* changelog name [skip ci]",
          "timestamp": "2026-08-24T13:47:06Z",
          "url": "https://github.com/gammasim/simtools/commit/0fe704ec183eb1ee61f5d52e5b06fde94ad7d2bb"
        },
        "date": 1787711095884,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 47.850864172,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 48.17999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 416.75,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 832.4486076720001,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 991.75,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1047.67578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 9.413899756999967,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 9.989999999999998,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 567.08203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 5.938535617999946,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.4799999999999995,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 429.16015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.438399687000015,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 7.9,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 439.515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.170934865999925,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 5.04,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 399.84765625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 32.96698010499995,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 26.48,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 410.78125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 6.770364703000041,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 7.3599999999999985,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 433.9453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 10.488997721000032,
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
            "value": 594.51171875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 14.107955360999995,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 14.649999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 694.16796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 7.816014827999993,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 8.559999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 425.6875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 8.020158652000077,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 8.49,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 457.12890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.828358770000023,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 5.92,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 598.7578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.055617694000034,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.339999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 714.0859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 10.54739505300006,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 9.7,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 519.73828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 16.951616976000082,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 17.36,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 628.9921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 12.211842131000026,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 12.87,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 614.140625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.03766374700001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 15.36,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 387.15625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 8.70502563499997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 13.9,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 734.71875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.719288548999998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.160000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 730.33203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 18.541416161000143,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 21.97,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1047.67578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.337954326000045,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 5.98,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 592.55078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 35.606275720999975,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 40.53,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1046.828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 10.98139477399991,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 12.860000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 851.9140625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 17.952531988000146,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 18.070000000000004,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 494.6796875,
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
          "id": "3a50d45820073017bde10cd430f3cd35c12db2e2",
          "message": "Improve production_derive_corsika_limits.py application. (#2470)\n\n* simplification for derivation and plotting of corsika limits\n\n* improve plotting and usable histograms\n\n* minimum trigger\n\n* improve plotting\n\n* unit tests\n\n* axis limits\n\n* plot reduction\n\n* simplification\n\n* reviewers comments\n\n* Replace hard-wired list of particle types for corsika limits derivations. (#2471)\n\n* remove hardcoded list of particles\n\n* changelog\n\n* code improvements\n\n* reviewers comments",
          "timestamp": "2026-08-26T16:35:46+02:00",
          "tree_id": "5e282b2ddf57782ad761b133623d2f1f48010a2f",
          "url": "https://github.com/gammasim/simtools/commit/3a50d45820073017bde10cd430f3cd35c12db2e2"
        },
        "date": 1787755620858,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 56.570544759,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 58.2,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 417.0546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 530.803393324,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 675.61,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1049.18359375,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 5.226436731999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 5.68,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 568.1484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 20.935117972,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 16.59,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 411.8828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 5.650154133000001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 6.160000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 597.8203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 7.945129262000023,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 8.54,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 695.90625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 7.063947105000011,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 6.44,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 539.9140625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 11.469168790000026,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 11.97,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 631.19921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 8.272461091999958,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 8.73,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 615.52734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 10.072506540999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 10.350000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 391.76953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 5.7980443010000045,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 8.64,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 699.4375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 11.169953335000002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 13.24,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1045.2578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 22.34715311299999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 25.520000000000003,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1049.18359375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 6.279553945000032,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 7.5,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 846.1875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 8.286087511000005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 8.48,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 494.85546875,
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
          "id": "9ca595a2668265709fe54f3bd3cc999cc5141ed9",
          "message": "Consistent version naming (#2466)\n\n* Improved version usage\n\n* tag vs version\n\n* remove duplicated legacy code\n\n* allow legacy sim_telarray metadata\n\n* unit test\n\n* changelog\n\n* reviewers comments\n\n* remove nonsense\n\n* sonar\n\n* More consistent naming of version / tag\n\n* changelog",
          "timestamp": "2026-08-27T08:38:59+02:00",
          "tree_id": "a1da07f7fded3128de80509b08a0e8dd6b3d5feb",
          "url": "https://github.com/gammasim/simtools/commit/9ca595a2668265709fe54f3bd3cc999cc5141ed9"
        },
        "date": 1787813846141,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 56.018322778,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 58.04,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 417.48046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 647.526762022,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 817.01,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1050.73828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 6.705245325999982,
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
            "value": 570.703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 5.286778491000007,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 5.8100000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 436.73828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 26.811865095,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 21.759999999999998,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 413.73046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 6.842235372000005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 7.35,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 595.328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 10.171411809999995,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 10.670000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 662.77734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 5.086355266999988,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 5.91,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 419.9296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 5.5158770140000115,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 6.09,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 716.40234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 9.166035684000008,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 8.61,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 510.828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 15.084937833000026,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 15.78,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 633.41796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 10.697793912000009,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 11.219999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 619.61328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 12.098991411999975,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 12.399999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 393.0859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 6.990594358999999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 10.88,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 735.4765625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 5.13874898499995,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 5.630000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 731,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 13.629580335000014,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 16.22,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1045.328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 28.90319390100001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 33.11,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1050.73828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 7.9673972429999935,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 9.51,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 843.89453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 11.583065446999967,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 11.82,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 497.64453125,
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
          "id": "9e6c9d89d7d6ade0772063613fd0c26fc141d66a",
          "message": "Fix in setting the environment (#2473)",
          "timestamp": "2026-08-27T11:38:36+02:00",
          "tree_id": "54e18813b0c070fcbed3742ff29ba08de4538205",
          "url": "https://github.com/gammasim/simtools/commit/9e6c9d89d7d6ade0772063613fd0c26fc141d66a"
        },
        "date": 1787824490172,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 43.454893111000004,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 42.8,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 416.3203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 851.922518106,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 1001.1,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1049.55859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 9.201106929999995,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 9.75,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 569.82421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 6.020921068999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.429999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 428.703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.260473134999984,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 7.720000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 435.26171875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.316574908999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 5.14,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 399.5625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 35.130047058,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 28.450000000000003,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 412.12109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 6.480982782000012,
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
            "value": 431.6484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 9.572685461999981,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 10.120000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 597.09765625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 13.39234714600002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 14.000000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 663.4765625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 6.887665838999965,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 7.68,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 424.77734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 5.125099294999984,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 5.640000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 441.96875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.685418261999985,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 5.830000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 622.2265625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.108066925000003,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.4799999999999995,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 721.83203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 11.25485474200002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 10.44,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 540.9375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 18.21422469800001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 19.04,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 632.01953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 13.085823151,
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
            "value": 617.30078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.67362792299997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 15.889999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 390.8359375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 9.142834007999966,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 14.040000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 698.70703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.672501006999994,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.0200000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 729.40625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 19.164146551999977,
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
            "value": 1046.65234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.220921429999976,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 5.53,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 592.28125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 37.11624184300001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 42.4,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1049.55859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 11.083394064000004,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 13.21,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 857.8046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 15.245773751999991,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 15.289999999999997,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 495.9453125,
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
          "id": "9e6c9d89d7d6ade0772063613fd0c26fc141d66a",
          "message": "Fix in setting the environment (#2473)",
          "timestamp": "2026-08-27T09:38:36Z",
          "url": "https://github.com/gammasim/simtools/commit/9e6c9d89d7d6ade0772063613fd0c26fc141d66a"
        },
        "date": 1787826217698,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 55.140022972,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 57.050000000000004,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 417.5234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260816.277.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 797.879363362,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 811.23,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1046.4375,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 7.028431264000005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 7.459999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 570.5390625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-ctao-array-layouts_run / wall_time_s",
            "value": 136.69430235199997,
            "unit": "wall_time_s",
            "extra": "outcome=skipped | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-ctao-array-layouts_run / cpu_time_s",
            "value": 2.24,
            "unit": "cpu_time_s",
            "extra": "outcome=skipped | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-ctao-array-layouts_run / peak_rss_mib",
            "value": 337.74609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=skipped | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 5.910409694000009,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 6.35,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 432.44140625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 27.719654369000068,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 22.31,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 415.08984375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 5.017595061999941,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 5.640000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 433.953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 7.192379232999997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 7.81,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 595.59765625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 10.300561420999998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 10.75,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 664.171875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 5.312602029000004,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 6.000000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 423.80078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 5.484055374000036,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 5.98,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 713.10546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 8.76325543200005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 8.030000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 510.98828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 14.19469904899995,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 14.809999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 622.99609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 10.14621463200001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 10.72,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 614.49609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 12.127091127999961,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 12.65,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 431.8046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 7.1170404399999825,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 11.02,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 726.05078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 5.224574791000009,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 5.420000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 721.09375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 14.777280660000088,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 17.54,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1044.890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 28.40250352199996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 32.52,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1046.4375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 8.510691454000039,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 10.16,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 847.46875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 11.77564517899998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 11.979999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 493.19921875,
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
          "id": "5532eab6210fde4bc5b74ac9a92aa1af8d992a4d",
          "message": "Add vim and diff to developer container (#2474)",
          "timestamp": "2026-08-27T14:31:58+02:00",
          "tree_id": "fa3259d355d0745a0ee5175d1401783ce7af6883",
          "url": "https://github.com/gammasim/simtools/commit/5532eab6210fde4bc5b74ac9a92aa1af8d992a4d"
        },
        "date": 1787834851352,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 59.573971314000005,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 61.83,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 417.625,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 800.395774053,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 934.42,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1064.12109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 8.580744480999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 9.010000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 571.48828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 5.476903456000002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 5.92,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 428.29296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 6.756057494000004,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 7.23,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 433.5859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 30.406956176999984,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 24.119999999999997,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 410.734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 6.198792538000021,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 6.7,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 430.75,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 8.885654615999954,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 9.379999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 592.42578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 13.247538642999984,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 13.81,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 660.30859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 6.917087349999974,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 7.69,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 427.12109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.178710526999964,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 5.35,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 605.12109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 6.5568189240000265,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 6.7700000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 710.703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 10.125064454999972,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 9.64,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 540.796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 16.40987381800005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 16.91,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 639.4453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 11.813060237999991,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 12.309999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 631.08203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 14.188861336999992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 14.51,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 402.7890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 8.30779236799998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 12.01,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 667.3828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.210929168999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 6.54,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 739.72265625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 17.161277331000008,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 20.13,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1058.21875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 33.73529073200001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 38.47,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1064.12109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 10.314722082000003,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 11.940000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 849.85546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 14.614740846000018,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 14.530000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.5 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 496.03125,
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
          "id": "87e5b9cb42ad43de7f6151206fc5749583bbe137",
          "message": "Prepare test resources from template (#2477)\n\n* Fix in setting the environment\n\n* Prepare test resources from template\n\n* new test resources\n\n* changelog\n\n* printout failed integration tests\n\n* move to v0.37.0 simtools-tests resources\n\n* fix nsb setting\n\n* remove hardwired versions in unit tests\n\n* changelog\n\n* simtools-tests v0.37.0\n\n* Potential fix for pull request finding [skip ci]\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>\n\n* Potential fix for pull request finding\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>\n\n* ruff\n\n---------\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>",
          "timestamp": "2026-08-28T07:19:36Z",
          "url": "https://github.com/gammasim/simtools/commit/87e5b9cb42ad43de7f6151206fc5749583bbe137"
        },
        "date": 1787918189093,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 59.85085046600001,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 61.85,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 417.37109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 873.745894268,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 1035.91,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1050.0078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 9.545434446999991,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 10.03,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 582.62109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 6.3500132670000085,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.96,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 428.14453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.71341950499999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 8.22,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 441.05859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.562472829000001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 5.27,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 397.98828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 33.78301296000001,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 27.03,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 412.78125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 7.015163362999999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 7.5600000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 433.23046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 10.781546297999967,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 11.37,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 594.83203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 14.065709529999992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 14.69,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 712.14453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 8.359482473000014,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 9.01,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 435.12890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 5.535448744000007,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 6.14,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 439.1328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.956891452999969,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 6.1,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 614.796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.299887506000005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.54,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 719.83203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 10.702628441000002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 10.15,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 515.7109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 17.396498476000033,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 18.03,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 628.4609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 12.269650021000018,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 12.879999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 618.12890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.18560532600003,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 15.8,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 435.859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 8.942529180999998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 13.52,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 696.45703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.924088428999994,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.3500000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 730.23828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 19.12819062999995,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 22.72,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1048.58984375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.51388103599993,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 6.000000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 591.64453125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 36.20508048199997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 41.26,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1050.0078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 11.185778870999911,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 13.160000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 851.3828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / wall_time_s",
            "value": 5.069898436000017,
            "unit": "wall_time_s",
            "extra": "outcome=skipped | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / cpu_time_s",
            "value": 5.3500000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=skipped | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / peak_rss_mib",
            "value": 449.203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=skipped | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 18.661213117999978,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 18.720000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 496.1953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
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
          "id": "5cd665c728a4153b287a7b58d5cde7ecc0c5dc7a",
          "message": "Update default simulation models to v0.17.1 (#2483)\n\n* Update default simulation models to v0.17.1\n\n* remove hardwired versions",
          "timestamp": "2026-08-28T17:17:30+02:00",
          "tree_id": "0c0e72b4db002cf391245fa12ff5820110fec481",
          "url": "https://github.com/gammasim/simtools/commit/5cd665c728a4153b287a7b58d5cde7ecc0c5dc7a"
        },
        "date": 1787931226365,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 45.86446266600001,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 45.21,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 414.09375,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 839.286677647,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 1001.08,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1049.69921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 9.290120330999997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 9.940000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 582.10546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 6.258902129999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.69,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 430.890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.617343622000021,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 8.04,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 440.1015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.188723716999959,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 5.130000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 399.71875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 31.799360413999977,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 25,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 411.65234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 7.026427162999994,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 7.6000000000000005,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 431.21875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 10.476900470000032,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 11.04,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 595.5625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 13.790330308000023,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 14.450000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 713.37890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 7.985504123999988,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 8.850000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 437.34765625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 5.23575659200003,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 5.739999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 439.9296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.77681512700002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 6.010000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 623.76171875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 6.844594216000019,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.3,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 718.53515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 10.509455435000007,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 9.84,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 538.84375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 17.086896531000036,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 17.89,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 630.2109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 12.194493485999942,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 12.75,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 614.6875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.065900608999982,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 15.160000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 388.51953125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 8.697555583000053,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 13.44,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 683.23046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.459273402000008,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 6.87,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 726.19921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 17.795110471000044,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 21.540000000000003,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1043.07421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.3216154430000415,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 5.880000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 590.41015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 34.763328459000036,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 39.71,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1049.69921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 10.725302864000014,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 12.75,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 850.15234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 18.000878552000017,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 18.130000000000003,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 493.6015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
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
          "id": "5cd665c728a4153b287a7b58d5cde7ecc0c5dc7a",
          "message": "Update default simulation models to v0.17.1 (#2483)\n\n* Update default simulation models to v0.17.1\n\n* remove hardwired versions",
          "timestamp": "2026-08-28T15:17:30Z",
          "url": "https://github.com/gammasim/simtools/commit/5cd665c728a4153b287a7b58d5cde7ecc0c5dc7a"
        },
        "date": 1787988637810,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 44.383577491,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 44.53,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 415.37109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 692.723189446,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 837.68,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1055.890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 7.116616594000021,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 7.62,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 585.2421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-ctao-array-layouts_run / wall_time_s",
            "value": 5.393384683000022,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-ctao-array-layouts_run / cpu_time_s",
            "value": 2.1599999999999997,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-ctao-array-layouts_run / peak_rss_mib",
            "value": 338.86328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 5.682883713999985,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 6.149999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 440.5546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 26.88448788300002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 20.97,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 412.47265625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 5.21732227199999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 5.79,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 432.19921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 7.454976239000018,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 7.970000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 595.36328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 9.906424937999986,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 10.55,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 713.6484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 5.339657373999955,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 5.98,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 420.25390625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 5.4158811350000065,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 5.79,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 711.41015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 8.811395568999956,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 7.99,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 513.09765625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 14.331549467999992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 15.1,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 637.33203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 10.127363191000029,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 10.67,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 627.5390625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 12.168356974000005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 12.44,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 396.9609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 7.007041092000009,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 10.790000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 738.05078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 5.107505512999978,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 5.5,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 734.80078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 14.356254146000026,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 17.259999999999998,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1053.87890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 28.29743851799998,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 32.6,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1055.890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 8.67389079000003,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 10.47,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 860.2265625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 11.81877160800002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 11.829999999999998,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 499.25,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
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
          "id": "5cd665c728a4153b287a7b58d5cde7ecc0c5dc7a",
          "message": "Update default simulation models to v0.17.1 (#2483)\n\n* Update default simulation models to v0.17.1\n\n* remove hardwired versions",
          "timestamp": "2026-08-28T15:17:30Z",
          "url": "https://github.com/gammasim/simtools/commit/5cd665c728a4153b287a7b58d5cde7ecc0c5dc7a"
        },
        "date": 1788069565001,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 51.07559902199999,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 52.12,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 414.19921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 995.6473475659999,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 1023.44,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1047.109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 9.457656390000011,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 10.120000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 583.98828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-ctao-array-layouts_run / wall_time_s",
            "value": 135.567538927,
            "unit": "wall_time_s",
            "extra": "outcome=skipped | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-ctao-array-layouts_run / cpu_time_s",
            "value": 2.6300000000000003,
            "unit": "cpu_time_s",
            "extra": "outcome=skipped | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-ctao-array-layouts_run / peak_rss_mib",
            "value": 339.2734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=skipped | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 6.338155952000022,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 6.88,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 433.03515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 7.701332303000015,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 8.129999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 441.93359375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.28237160599997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 5.07,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 398.390625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 32.26897219900002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 25.29,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 414.3046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 7.084972217000029,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 7.69,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 435.41796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 10.546723728000018,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 11.08,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 598.56640625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 13.693981257000019,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 14.18,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 714.90234375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 7.870575925000026,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 8.649999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 436.16796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 5.292582533999962,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 5.8999999999999995,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 442.1796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 5.89008552100006,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 6.01,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 608.53515625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 6.986681568999984,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 7.410000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 720.08984375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 10.608678634000057,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 9.690000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 519.7421875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 16.942474310999955,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 17.68,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 632.17578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 12.25117814400005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 12.57,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 616.9921875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 15.137092156999984,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 15.68,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 425.37890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 8.914189319000002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 13.540000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 695.48828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 6.62115570900005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 6.96,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 727.55078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 17.954905860000054,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 21.46,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1044.7265625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 5.554960767000011,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 6.159999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 591.3828125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 35.54664123099997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 40.62,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1047.109375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 10.791082016000018,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 12.77,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 860.0078125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 18.379260255999952,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 18.54,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 495.66796875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
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
          "id": "5cd665c728a4153b287a7b58d5cde7ecc0c5dc7a",
          "message": "Update default simulation models to v0.17.1 (#2483)\n\n* Update default simulation models to v0.17.1\n\n* remove hardwired versions",
          "timestamp": "2026-08-28T15:17:30Z",
          "url": "https://github.com/gammasim/simtools/commit/5cd665c728a4153b287a7b58d5cde7ecc0c5dc7a"
        },
        "date": 1788156850923,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "unit-session / wall_time_s",
            "value": 59.01190853499999,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / cpu_time_s",
            "value": 61.42,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "unit-session / peak_rss_mib",
            "value": 415.45703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=ubuntu24/20260823.283.1 | container=None | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / wall_time_s",
            "value": 973.0736072340001,
            "unit": "wall_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / cpu_time_s",
            "value": 1135.23,
            "unit": "cpu_time_s",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration-session / 7.0.0 / peak_rss_mib",
            "value": 1051.96875,
            "unit": "peak_rss_mib",
            "extra": "outcome=exit_status=0 | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / wall_time_s",
            "value": 10.616163697000005,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / cpu_time_s",
            "value": 11.23,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-compare-productions_run / peak_rss_mib",
            "value": 584.00390625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / wall_time_s",
            "value": 6.959019928999993,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / cpu_time_s",
            "value": 7.41,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots0 / peak_rss_mib",
            "value": 433.60546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / wall_time_s",
            "value": 8.581481889999992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / cpu_time_s",
            "value": 9.149999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-incident-angle_incident_angles_dual_mirror_debug_plots1 / peak_rss_mib",
            "value": 446.33984375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / wall_time_s",
            "value": 5.892716058000019,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / cpu_time_s",
            "value": 5.53,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-mirror-rnda_psf_measurement / peak_rss_mib",
            "value": 398.09375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / wall_time_s",
            "value": 35.13437754899999,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / cpu_time_s",
            "value": 27.26,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-derive-psf-parameters_run / peak_rss_mib",
            "value": 413.27734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-docs-produce-array-element-report_run / wall_time_s",
            "value": 5.197527716000025,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-docs-produce-array-element-report_run / cpu_time_s",
            "value": 5.83,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-docs-produce-array-element-report_run / peak_rss_mib",
            "value": 413.69140625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / wall_time_s",
            "value": 7.712410691000002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / cpu_time_s",
            "value": 8.249999999999998,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-generate-corsika-histograms_plot / peak_rss_mib",
            "value": 432.98046875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-array-layout_from_list_north / wall_time_s",
            "value": 5.52693624799997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-array-layout_from_list_north / cpu_time_s",
            "value": 5.99,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-array-layout_from_list_north / peak_rss_mib",
            "value": 427.8671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / wall_time_s",
            "value": 12.067190056000015,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / cpu_time_s",
            "value": 12.58,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simtel-events_flasher_event_plots / peak_rss_mib",
            "value": 595.29296875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / wall_time_s",
            "value": 15.424967890999994,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / cpu_time_s",
            "value": 16.02,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-simulated-event-distributions_sim_telarray_input / peak_rss_mib",
            "value": 714.12890625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-tabular-data-for-model-parameter_atmospheric_profile_all / wall_time_s",
            "value": 5.504540107000025,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-tabular-data-for-model-parameter_atmospheric_profile_all / cpu_time_s",
            "value": 5.96,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-tabular-data-for-model-parameter_atmospheric_profile_all / peak_rss_mib",
            "value": 405.77734375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / wall_time_s",
            "value": 8.88709911999996,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / cpu_time_s",
            "value": 9.620000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-production-generate-grid_production_generate_grid_horizontal_density / peak_rss_mib",
            "value": 438.5703125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / wall_time_s",
            "value": 6.013763528000027,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / cpu_time_s",
            "value": 6.680000000000001,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-plot-corsika-limits_plot_corsika_limits / peak_rss_mib",
            "value": 437.9609375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_lstn / wall_time_s",
            "value": 5.01401097400003,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_lstn / cpu_time_s",
            "value": 5.2799999999999985,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_lstn / peak_rss_mib",
            "value": 623.9375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / wall_time_s",
            "value": 6.5597456340000235,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / cpu_time_s",
            "value": 6.85,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direct_injection_lstn_filter_wheel_sequence / peak_rss_mib",
            "value": 627.1015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / wall_time_s",
            "value": 7.80075478599997,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / cpu_time_s",
            "value": 8.1,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_direction_injection_msfx_flashcam_south / peak_rss_mib",
            "value": 713.375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / wall_time_s",
            "value": 11.837891678999995,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / cpu_time_s",
            "value": 11.42,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_alpha_north / peak_rss_mib",
            "value": 542.80859375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / wall_time_s",
            "value": 18.765936778000025,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / cpu_time_s",
            "value": 19.55,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_different_intensities / peak_rss_mib",
            "value": 634.125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / wall_time_s",
            "value": 13.457302458999948,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / cpu_time_s",
            "value": 13.97,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-flasher_full_simulation_lst_filter_wheel_single_event_value / peak_rss_mib",
            "value": 619.58203125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / wall_time_s",
            "value": 16.48449427500009,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / cpu_time_s",
            "value": 16.82,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-layout / peak_rss_mib",
            "value": 443.86328125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / wall_time_s",
            "value": 9.751567789999967,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / cpu_time_s",
            "value": 14.979999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-illuminator_run-multi-illuminator / peak_rss_mib",
            "value": 684.41015625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / wall_time_s",
            "value": 7.383280047000085,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / cpu_time_s",
            "value": 7.869999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-pedestals_pedestals_20_deg_north / peak_rss_mib",
            "value": 728.81640625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / wall_time_s",
            "value": 20.01378580300002,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / cpu_time_s",
            "value": 23.810000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_20_deg_multiple_model_versions / peak_rss_mib",
            "value": 1045.92578125,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / wall_time_s",
            "value": 6.047923775000072,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / cpu_time_s",
            "value": 6.519999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_40_deg_south_corsika_only / peak_rss_mib",
            "value": 591.65625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / wall_time_s",
            "value": 38.17792704599992,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / cpu_time_s",
            "value": 43.93,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_gamma_62_deg_south_check_output / peak_rss_mib",
            "value": 1051.96875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / wall_time_s",
            "value": 12.194363582999927,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / cpu_time_s",
            "value": 14.449999999999998,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-simulate-prod_proton_20_deg_north_check_output / peak_rss_mib",
            "value": 851.63671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_MSTS / wall_time_s",
            "value": 5.58539281700007,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_MSTS / cpu_time_s",
            "value": 6.000000000000002,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_MSTS / peak_rss_mib",
            "value": 449.09765625,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_MSTN / wall_time_s",
            "value": 5.505661889999942,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_MSTN / cpu_time_s",
            "value": 5.819999999999999,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_MSTN / peak_rss_mib",
            "value": 448.6484375,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / wall_time_s",
            "value": 5.687588796999989,
            "unit": "wall_time_s",
            "extra": "outcome=skipped | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / cpu_time_s",
            "value": 5.94,
            "unit": "cpu_time_s",
            "extra": "outcome=skipped | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-camera-efficiency_SSTS / peak_rss_mib",
            "value": 450.3671875,
            "unit": "peak_rss_mib",
            "extra": "outcome=skipped | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / wall_time_s",
            "value": 19.799428595999984,
            "unit": "wall_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / cpu_time_s",
            "value": 19.710000000000004,
            "unit": "cpu_time_s",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          },
          {
            "name": "integration / 7.0.0 / simtools-validate-optics_run / peak_rss_mib",
            "value": 495.10546875,
            "unit": "peak_rss_mib",
            "extra": "outcome=passed | python=3.14.7 | runner=Linux/X64 | runner_image=None/None | container=ghcr.io/gammasim/simtools-dev:latest | sample_interval_s=0.2"
          }
        ]
      }
    ]
  }
}