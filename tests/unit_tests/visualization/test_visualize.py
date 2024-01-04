#!/usr/bin/python3

import logging

import astropy.io.ascii
import astropy.units as u
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pytest

import simtools.utils.general as gen
from simtools.visualization import visualize

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_plot_1d(db, io_handler):
    logger.debug("Testing plot_1d")

    x_title = "Wavelength [nm]"
    y_title = "Mirror reflectivity [%]"
    headers_type = {"names": (x_title, y_title), "formats": ("f8", "f8")}
    title = "Test 1D plot"

    test_file_name = "ref_200_1100_190211a.dat"
    db.export_file_db(
        db_name=db.DB_CTA_SIMULATION_MODEL,
        dest=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
        file_name=test_file_name,
    )
    test_data_file = gen.find_file(
        test_file_name,
        io_handler.get_output_directory(sub_dir="model", dir_type="test"),
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

    plt = visualize.plot_1d(data, title=title, palette="autumn")

    plot_file = io_handler.get_output_file(
        file_name="plot_1d.pdf", sub_dir="plots", dir_type="test"
    )
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
        dest=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
        file_name=test_file_name,
    )
    table_file = gen.find_file(
        test_file_name,
        io_handler.get_output_directory(sub_dir="model", dir_type="test"),
    )
    table = astropy.io.ascii.read(table_file)

    plt = visualize.plot_table(table, y_title="Transmission", title=title, no_markers=True)

    plot_file = io_handler.get_output_file(
        file_name="plot_table.pdf", sub_dir="plots", dir_type="test"
    )
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
    array_layout_north_instance,
    telescope_south_test_file,
    array_layout_south_instance,
):
    def test_one_site(test_file, instance):
        telescope_table = instance.initialize_array_layout_from_telescope_file(test_file)
        telescopes_dict = instance.include_radius_into_telescope_table(telescope_table)
        fig_out = visualize.plot_array(telescopes_dict, rotate_angle=0 * u.deg)
        assert isinstance(fig_out, type(plt.figure()))
        plt.close()

    test_one_site(telescope_north_test_file, array_layout_north_instance)
    test_one_site(telescope_south_test_file, array_layout_south_instance)
