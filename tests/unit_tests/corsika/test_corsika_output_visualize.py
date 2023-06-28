import matplotlib.pyplot as plt
import numpy as np
import pytest

from simtools.corsika import corsika_output_visualize


def test_kernel_plot_2D_photons(corsika_output_instance_set_histograms, caplog):
    corsika_output_instance_set_histograms.set_histograms(
        individual_telescopes=False, telescope_indices=[0, 1, 2]
    )
    for property_name in [
        "counts",
        "density",
        "direction",
        "time_altitude",
        "num_photons_per_telescope",
    ]:
        all_figs = corsika_output_visualize._kernel_plot_2D_photons(
            corsika_output_instance_set_histograms, property_name
        )
        assert np.size(all_figs) == 1
        assert isinstance(all_figs[0], type(plt.figure()))

    corsika_output_instance_set_histograms.set_histograms(
        individual_telescopes=True, telescope_indices=[0, 1, 2]
    )
    for property_name in ["counts", "density", "direction", "time_altitude"]:
        all_figs = corsika_output_visualize._kernel_plot_2D_photons(
            corsika_output_instance_set_histograms, property_name
        )
        for i_hist, _ in enumerate(corsika_output_instance_set_histograms.telescope_indices):
            assert isinstance(all_figs[i_hist], plt.Figure)

    with pytest.raises(ValueError):
        corsika_output_visualize._kernel_plot_2D_photons(
            corsika_output_instance_set_histograms, "this_property_does_not_exist"
        )
        msg = "results: status must be one of "
        assert msg in caplog.text


def test_plot_2Ds(corsika_output_instance_set_histograms):
    for function_label in [
        "plot_2D_counts",
        "plot_2D_density",
        "plot_2D_direction",
        "plot_2D_num_photons_per_telescope",
    ]:
        function = getattr(corsika_output_visualize, function_label)
        figs = function(corsika_output_instance_set_histograms)
        assert isinstance(figs, list)
        assert all(isinstance(fig, plt.Figure) for fig in figs)


def test_kernel_plot_1D_photons(corsika_output_instance_set_histograms, caplog):
    corsika_output_instance_set_histograms.set_histograms(
        individual_telescopes=False, telescope_indices=[0, 1, 2]
    )
    labels = ["wavelength", "counts", "density", "time", "altitude"]

    for property_name in np.append(labels, "num_photons"):
        all_figs = corsika_output_visualize._kernel_plot_1D_photons(
            corsika_output_instance_set_histograms, property_name
        )
        assert np.size(all_figs) == 1
        assert isinstance(all_figs[0], type(plt.figure()))

    corsika_output_instance_set_histograms.set_histograms(
        individual_telescopes=True, telescope_indices=[0, 1, 2]
    )
    for property_name in labels:
        all_figs = corsika_output_visualize._kernel_plot_1D_photons(
            corsika_output_instance_set_histograms, property_name
        )
        for i_hist, _ in enumerate(corsika_output_instance_set_histograms.telescope_indices):
            assert isinstance(all_figs[i_hist], plt.Figure)

    with pytest.raises(ValueError):
        corsika_output_visualize._kernel_plot_1D_photons(
            corsika_output_instance_set_histograms, "this_property_does_not_exist"
        )
        msg = "results: status must be one of "
        assert msg in caplog.text


def test_plot_1Ds(corsika_output_instance_set_histograms):
    for function_label in [
        "plot_wavelength_distr",
        "plot_counts_distr",
        "plot_density_distr",
        "plot_time_distr",
        "plot_altitude_distr",
        "plot_num_photons_distr",
    ]:
        function = getattr(corsika_output_visualize, function_label)
        figs = function(corsika_output_instance_set_histograms)
        assert isinstance(figs, list)
        assert all(isinstance(fig, plt.Figure) for fig in figs)
