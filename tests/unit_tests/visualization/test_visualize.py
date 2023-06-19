#!/usr/bin/python3

import logging

import astropy.io.ascii
import astropy.units as u
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pytest

import simtools.util.general as gen
from simtools.visualization import visualize

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_plot_1D(db, io_handler):

    logger.debug("Testing plot_1D")

    x_title = "Wavelength [nm]"
    y_title = "Mirror reflectivity [%]"
    headers_type = {"names": (x_title, y_title), "formats": ("f8", "f8")}
    title = "Test 1D plot"

    test_file_name = "ref_200_1100_190211a.dat"
    db.export_file_db(
        db_name=db.DB_CTA_SIMULATION_MODEL,
        dest=io_handler.get_output_directory(dir_type="model", test=True),
        file_name=test_file_name,
    )
    test_data_file = gen.find_file(
        test_file_name, io_handler.get_output_directory(dir_type="model", test=True)
    )
    data_in = np.loadtxt(test_data_file, usecols=(0, 1), dtype=headers_type)

    # Change y-axis to percent
    if "%" in y_title:
        if np.max(data_in[y_title]) <= 1:
            data_in[y_title] = 100 * data_in[y_title]
    data = dict()
    data["Reflectivity"] = data_in
    for i in range(5):
        new_data = np.copy(data_in)
        new_data[y_title] = new_data[y_title] * (1 - 0.1 * (i + 1))
        data[f"{100 * (1 - 0.1 * (i + 1))}%% reflectivity"] = new_data

    plt = visualize.plot_1D(data, title=title, palette="autumn")

    plot_file = io_handler.get_output_file(file_name="plot_1D.pdf", dir_type="plots", test=True)
    if plot_file.exists():
        plot_file.unlink()
    plt.savefig(plot_file)

    logger.debug(f"Produced 1D plot ({plot_file}).")

    assert plot_file.exists()


def test_plot_table(db, io_handler):

    logger.debug("Testing plot_table")

    title = "Test plot table"

    test_file_name = "Transmission_Spectrum_PlexiGlass.dat"
    db.export_file_db(
        db_name="test-data",
        dest=io_handler.get_output_directory(dir_type="model", test=True),
        file_name=test_file_name,
    )
    table_file = gen.find_file(
        test_file_name, io_handler.get_output_directory(dir_type="model", test=True)
    )
    table = astropy.io.ascii.read(table_file)

    plt = visualize.plot_table(table, y_title="Transmission", title=title, no_markers=True)

    plot_file = io_handler.get_output_file(file_name="plot_table.pdf", dir_type="plots", test=True)
    if plot_file.exists():
        plot_file.unlink()
    plt.savefig(plot_file)

    logger.debug(f"Produced 1D plot ({plot_file}).")

    assert plot_file.exists()


def test_add_unit():

    value_with_unit = [30, 40] << u.nm
    assert visualize._add_unit("Wavelength", value_with_unit) == "Wavelength [nm]"
    value_without_unit = [30, 40]
    assert visualize._add_unit("Wavelength", value_without_unit) == "Wavelength"


def test_get_telescope_patch(manual_corsika_dict_north, manual_corsika_dict_south, io_handler):
    def test_one_site(corsika_dict, x, y):
        for tel_type in np.array(list(corsika_dict["corsika_sphere_radius"].keys())):
            radius = corsika_dict["corsika_sphere_radius"][tel_type].value
            patch = visualize.get_telescope_patch(tel_type, x, y, radius * u.m)
            if mpatches.Circle == type(patch):
                assert patch.radius == corsika_dict["corsika_sphere_radius"][tel_type].value
            else:
                assert isinstance(patch, mpatches.Rectangle)

    test_one_site(manual_corsika_dict_north, 0 * u.m, 0 * u.m)
    test_one_site(manual_corsika_dict_south, 0 * u.m, 0 * u.m)
    # Test passing other units
    test_one_site(manual_corsika_dict_north, 0 * u.m, 0 * u.km)
    test_one_site(manual_corsika_dict_south, 0 * u.cm, 0 * u.km)
    with pytest.raises(TypeError):
        test_one_site(manual_corsika_dict_south, 0, 0)


def test_plot_array(
    telescope_north_test_file,
    layout_array_north_instance,
    telescope_south_test_file,
    layout_array_south_instance,
):
    def test_one_site(test_file, instance):
        telescope_table = instance.read_telescope_list_file(test_file)
        telescopes_dict = instance.include_radius_into_telescope_table(telescope_table)
        fig_out = visualize.plot_array(telescopes_dict, rotate_angle=0 * u.deg)
        assert isinstance(fig_out, type(plt.figure()))

    test_one_site(telescope_north_test_file, layout_array_north_instance)
    test_one_site(telescope_south_test_file, layout_array_south_instance)


def test_kernel_plot_2D_photons(corsika_output_instance_set_histograms, caplog):
    corsika_output_instance_set_histograms.set_histograms(individual_telescopes=False)
    for property_name in [
        "counts",
        "density",
        "direction",
        "time_altitude",
        "num_photons_per_telescope",
    ]:
        all_figs = visualize._kernel_plot_2D_photons(
            corsika_output_instance_set_histograms, property_name
        )
        assert np.size(all_figs) == 1
        assert isinstance(all_figs[0], type(plt.figure()))

    corsika_output_instance_set_histograms.set_histograms(
        individual_telescopes=True, telescope_indices=[0, 1, 2]
    )
    for property_name in ["counts", "density", "direction", "time_altitude"]:
        all_figs = visualize._kernel_plot_2D_photons(
            corsika_output_instance_set_histograms, property_name
        )
        for i_hist, _ in enumerate(corsika_output_instance_set_histograms.telescope_indices):
            assert isinstance(all_figs[i_hist], plt.Figure)

    with pytest.raises(ValueError):
        all_figs = visualize._kernel_plot_2D_photons(
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
        function = getattr(visualize, function_label)
        figs = function(corsika_output_instance_set_histograms)
        assert isinstance(figs, list)
        assert all(isinstance(fig, plt.Figure) for fig in figs)


def test_kernel_plot_1D_photons(corsika_output_instance_set_histograms, caplog):
    corsika_output_instance_set_histograms.set_histograms(individual_telescopes=False)
    labels = ["wavelength", "counts", "density", "time", "altitude"]

    for property_name in np.append(labels, "num_photons"):
        all_figs = visualize._kernel_plot_1D_photons(
            corsika_output_instance_set_histograms, property_name
        )
        assert np.size(all_figs) == 1
        assert isinstance(all_figs[0], type(plt.figure()))

    corsika_output_instance_set_histograms.set_histograms(
        individual_telescopes=True, telescope_indices=[0, 1, 2]
    )
    for property_name in labels:
        all_figs = visualize._kernel_plot_1D_photons(
            corsika_output_instance_set_histograms, property_name
        )
        for i_hist, _ in enumerate(corsika_output_instance_set_histograms.telescope_indices):
            assert isinstance(all_figs[i_hist], plt.Figure)

    with pytest.raises(ValueError):
        all_figs = visualize._kernel_plot_1D_photons(
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
        function = getattr(visualize, function_label)
        figs = function(corsika_output_instance_set_histograms)
        assert isinstance(figs, list)
        assert all(isinstance(fig, plt.Figure) for fig in figs)
