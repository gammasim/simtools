import matplotlib.pyplot as plt
import numpy as np
import pytest

from simtools.corsika import corsika_histograms_visualize


def test_kernel_plot_2d_photons(corsika_histograms_instance_set_histograms, caplog):
    corsika_histograms_instance_set_histograms.set_histograms(
        individual_telescopes=False, telescope_indices=[0, 1, 2]
    )
    for property_name in [
        "counts",
        "density",
        "direction",
        "time_altitude",
        "num_photons_per_telescope",
    ]:
        all_figs = corsika_histograms_visualize._kernel_plot_2d_photons(
            corsika_histograms_instance_set_histograms, property_name
        )
        assert np.size(all_figs) == 1
        assert isinstance(all_figs[0], type(plt.figure()))

    corsika_histograms_instance_set_histograms.set_histograms(
        individual_telescopes=True, telescope_indices=[0, 1, 2]
    )
    for property_name in [
        "counts",
        "density",
        "direction",
        "time_altitude",
        "num_photons_per_telescope",
    ]:
        all_figs = corsika_histograms_visualize._kernel_plot_2d_photons(
            corsika_histograms_instance_set_histograms, property_name
        )
        for _, _ in enumerate(corsika_histograms_instance_set_histograms.telescope_indices):
            assert isinstance(all_figs[0], plt.Figure)

    with pytest.raises(ValueError):
        corsika_histograms_visualize._kernel_plot_2d_photons(
            corsika_histograms_instance_set_histograms, "this_property_does_not_exist"
        )
        msg = "This property does not exist. "
        assert msg in caplog.text


def test_plot_2ds(corsika_histograms_instance_set_histograms):
    for function_label in [
        "plot_2d_counts",
        "plot_2d_density",
        "plot_2d_direction",
        "plot_2d_num_photons_per_telescope",
    ]:
        function = getattr(corsika_histograms_visualize, function_label)
        figs = function(corsika_histograms_instance_set_histograms)
        assert isinstance(figs, list)
        assert all(isinstance(fig, plt.Figure) for fig in figs)


def test_kernel_plot_1d_photons(corsika_histograms_instance_set_histograms, caplog):
    corsika_histograms_instance_set_histograms.set_histograms(
        individual_telescopes=False, telescope_indices=[0, 1, 2]
    )
    labels = [
        "wavelength",
        "counts",
        "density",
        "time",
        "altitude",
        "num_photons_per_event",
        "num_photons_per_telescope",
    ]

    for property_name in labels:
        all_figs = corsika_histograms_visualize._kernel_plot_1d_photons(
            corsika_histograms_instance_set_histograms, property_name
        )
        assert np.size(all_figs) == 1
        assert isinstance(all_figs[0], type(plt.figure()))

    corsika_histograms_instance_set_histograms.set_histograms(
        individual_telescopes=True, telescope_indices=[0, 1, 2]
    )
    for property_name in labels:
        all_figs = corsika_histograms_visualize._kernel_plot_1d_photons(
            corsika_histograms_instance_set_histograms, property_name
        )
        for i_hist, _ in enumerate(corsika_histograms_instance_set_histograms.telescope_indices):
            if property_name in ["num_photons_per_event", "num_photons_per_telescope"]:
                assert isinstance(all_figs[0], plt.Figure)
            else:
                assert isinstance(all_figs[i_hist], plt.Figure)

    with pytest.raises(ValueError):
        corsika_histograms_visualize._kernel_plot_1d_photons(
            corsika_histograms_instance_set_histograms, "this_property_does_not_exist"
        )
        msg = "This property does not exist. "
        assert msg in caplog.text


def test_plot_1ds(corsika_histograms_instance_set_histograms):
    for function_label in [
        "plot_wavelength_distr",
        "plot_counts_distr",
        "plot_density_distr",
        "plot_time_distr",
        "plot_altitude_distr",
        "plot_photon_per_event_distr",
        "plot_photon_per_telescope_distr",
    ]:
        function = getattr(corsika_histograms_visualize, function_label)
        figs = function(corsika_histograms_instance_set_histograms)
        assert isinstance(figs, list)
        assert all(isinstance(fig, plt.Figure) for fig in figs)


def test_plot_event_headers(corsika_histograms_instance_set_histograms):
    fig = corsika_histograms_visualize.plot_1d_event_header_distribution(
        corsika_histograms_instance_set_histograms, "total_energy"
    )
    assert isinstance(fig, plt.Figure)

    fig = corsika_histograms_visualize.plot_2d_event_header_distribution(
        corsika_histograms_instance_set_histograms, "zenith", "azimuth"
    )
    assert isinstance(fig, plt.Figure)


def test_save_figs_to_pdf(corsika_histograms_instance_set_histograms, io_handler):
    output_file = io_handler.get_output_directory(dir_type="test").joinpath("test.pdf")
    figs_list = []
    for function_label in [
        "plot_photon_per_event_distr",
        "plot_photon_per_telescope_distr",
    ]:
        function = getattr(corsika_histograms_visualize, function_label)
        figs = function(corsika_histograms_instance_set_histograms)
        figs_list.append(figs)
    figs_list = np.array(figs_list).flatten()
    corsika_histograms_visualize.save_figs_to_pdf(figs_list, output_file)
    assert output_file.exists()
