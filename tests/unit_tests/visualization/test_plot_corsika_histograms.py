import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u

from simtools.visualization import plot_corsika_histograms

# Ignore UserWarning (e.g., SciPy NumPy-version warning) at module level so pytest
# does not treat it as an error during collection.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

# Prevent RuntimeWarning from matplotlib about too many open figures during tests
mpl.rcParams["figure.max_open_warning"] = 0


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
        all_figs = plot_corsika_histograms._kernel_plot_2d_photons(
            corsika_histograms_instance_set_histograms, property_name
        )
        assert np.size(all_figs) == 1
        assert isinstance(all_figs[0], plt.Figure)

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
        all_figs = plot_corsika_histograms._kernel_plot_2d_photons(
            corsika_histograms_instance_set_histograms, property_name
        )
        for _, _ in enumerate(corsika_histograms_instance_set_histograms.telescope_indices):
            assert isinstance(all_figs[0], plt.Figure)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match=r"This property does not exist. The valid entries"):
            plot_corsika_histograms._kernel_plot_2d_photons(
                corsika_histograms_instance_set_histograms, "this_property_does_not_exist"
            )
    assert "This property does not exist. " in caplog.text


def test_plot_2ds(corsika_histograms_instance_set_histograms):
    for function_label in [
        "plot_2d_counts",
        "plot_2d_density",
        "plot_2d_direction",
        "plot_2d_num_photons_per_telescope",
    ]:
        _function = getattr(plot_corsika_histograms, function_label)
        figs = _function(corsika_histograms_instance_set_histograms)
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
        all_figs = plot_corsika_histograms._kernel_plot_1d_photons(
            corsika_histograms_instance_set_histograms, property_name
        )
        assert np.size(all_figs) == 1
        assert isinstance(all_figs[0], type(plt.figure()))

    corsika_histograms_instance_set_histograms.set_histograms(
        individual_telescopes=True, telescope_indices=[0, 1, 2]
    )
    for property_name in labels:
        all_figs = plot_corsika_histograms._kernel_plot_1d_photons(
            corsika_histograms_instance_set_histograms, property_name
        )
        for i_hist, _ in enumerate(corsika_histograms_instance_set_histograms.telescope_indices):
            if property_name in ["num_photons_per_event", "num_photons_per_telescope"]:
                assert isinstance(all_figs[0], plt.Figure)
            else:
                assert isinstance(all_figs[i_hist], plt.Figure)
    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError, match=r"This property does not"):
            plot_corsika_histograms._kernel_plot_1d_photons(
                corsika_histograms_instance_set_histograms, "this_property_does_not_exist"
            )
    assert "This property does not exist. " in caplog.text


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
        _function = getattr(plot_corsika_histograms, function_label)
        figs = _function(corsika_histograms_instance_set_histograms)
        assert isinstance(figs, list)
        assert all(isinstance(fig, plt.Figure) for fig in figs)


def test_plot_event_headers(corsika_histograms_instance_set_histograms):
    fig = plot_corsika_histograms.plot_1d_event_header_distribution(
        corsika_histograms_instance_set_histograms, "total_energy"
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_corsika_histograms.plot_2d_event_header_distribution(
        corsika_histograms_instance_set_histograms, "zenith", "azimuth"
    )
    assert isinstance(fig, plt.Figure)


def test_save_figs_to_pdf(corsika_histograms_instance_set_histograms, io_handler):
    output_file = io_handler.get_output_directory().joinpath("test.pdf")
    figs_list = []
    for function_label in [
        "plot_photon_per_event_distr",
        "plot_photon_per_telescope_distr",
    ]:
        _function = getattr(plot_corsika_histograms, function_label)
        figs = _function(corsika_histograms_instance_set_histograms)
        figs_list.append(figs)
    figs_list = np.array(figs_list).flatten()
    plot_corsika_histograms.save_figs_to_pdf(figs_list, output_file)
    assert output_file.exists()


def test_event_header_1d_dimensionless_and_log_toggle(corsika_histograms_instance_set_histograms):
    # Force dimensionless unit to exercise the else-branch for xlabel and log_y=False path
    key = "total_energy"
    corsika_histograms_instance_set_histograms.event_information[key] = (
        corsika_histograms_instance_set_histograms.event_information[key].value
        * u.dimensionless_unscaled
    )
    fig = plot_corsika_histograms.plot_1d_event_header_distribution(
        corsika_histograms_instance_set_histograms, key, log_y=False
    )
    assert isinstance(fig, plt.Figure)


def test_event_header_2d_logz_false_and_dimensionless_labels(
    corsika_histograms_instance_set_histograms,
):
    # Force both units to be dimensionless to cover xlabel/ylabel else-branches and log_z=False
    key_x = "zenith"
    key_y = "azimuth"
    corsika_histograms_instance_set_histograms.event_information[key_x] = (
        corsika_histograms_instance_set_histograms.event_information[key_x].value
        * u.dimensionless_unscaled
    )
    corsika_histograms_instance_set_histograms.event_information[key_y] = (
        corsika_histograms_instance_set_histograms.event_information[key_y].value
        * u.dimensionless_unscaled
    )
    fig = plot_corsika_histograms.plot_2d_event_header_distribution(
        corsika_histograms_instance_set_histograms, key_x, key_y, log_z=False
    )
    assert isinstance(fig, plt.Figure)


def test_build_and_export_all_photon_figures(
    corsika_histograms_instance_set_histograms, io_handler
):
    # Build a reduced set for speed
    figs = plot_corsika_histograms.build_all_photon_figures(
        corsika_histograms_instance_set_histograms, test=True
    )
    assert isinstance(figs, np.ndarray)
    assert figs.size > 0

    # Export reduced set to a single PDF
    pdf_path = plot_corsika_histograms.export_all_photon_figures_pdf(
        corsika_histograms_instance_set_histograms, test=True
    )
    assert pdf_path.exists()


def test_derive_event_histograms_pdf_and_hdf5(
    corsika_histograms_instance_set_histograms, io_handler
):
    # 1D event histograms: create PDF and HDF5 outputs
    pdf_1d = plot_corsika_histograms.derive_event_1d_histograms(
        corsika_histograms_instance_set_histograms,
        event_1d_header_keys=["total_energy", "zenith"],
        pdf=True,
        hdf5=True,
        overwrite=True,
    )
    assert pdf_1d is not None
    assert pdf_1d.exists()

    # 2D event histograms: create PDF and HDF5 outputs
    pdf_2d = plot_corsika_histograms.derive_event_2d_histograms(
        corsika_histograms_instance_set_histograms,
        event_2d_header_keys=["zenith", "azimuth"],
        pdf=True,
        hdf5=True,
        overwrite=True,
    )
    assert pdf_2d is not None
    assert pdf_2d.exists()


def test_derive_event_2d_histograms_odd_keys_warning(
    corsika_histograms_instance_set_histograms, caplog
):
    with caplog.at_level(logging.WARNING):
        # Use odd number of keys and disable outputs to only exercise the warning path
        result = plot_corsika_histograms.derive_event_2d_histograms(
            corsika_histograms_instance_set_histograms,
            event_2d_header_keys=["zenith", "azimuth", "total_energy"],
            pdf=False,
            hdf5=False,
            overwrite=False,
        )
    assert result is None
    assert "An odd number of keys was passed" in caplog.text


def test_derive_event_1d_histograms_no_pdf_returns_none(corsika_histograms_instance_set_histograms):
    # pdf disabled -> function should return None regardless of hdf5 export
    result = plot_corsika_histograms.derive_event_1d_histograms(
        corsika_histograms_instance_set_histograms,
        event_1d_header_keys=["total_energy"],
        pdf=False,
        hdf5=True,
        overwrite=False,
    )
    assert result is None


def test_plot_2d_time_altitude_returns_figs(corsika_histograms_instance_set_histograms):
    figs = plot_corsika_histograms.plot_2d_time_altitude(
        corsika_histograms_instance_set_histograms, log_z=True
    )
    assert isinstance(figs, list)
    assert len(figs) > 0
    for f in figs:
        # matplotlib Figure has savefig attribute
        assert hasattr(f, "savefig")
        f.clf()
