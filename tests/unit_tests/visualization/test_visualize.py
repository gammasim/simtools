#!/usr/bin/python3

import logging

import astropy.io.ascii
import astropy.units as u
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pytest

import simtools.utils.general as gen
from simtools.utils import names
from simtools.visualization import visualize

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_plot_1d(db, io_handler):
    logger.debug("Testing plot_1d")

    x_title = "Wavelength [nm]"
    y_title = "Mirror reflectivity [%]"
    headers_type = {"names": (x_title, y_title), "formats": ("f8", "f8")}
    title = "Test 1D plot"

    test_file_name = "ref_LST1_2022_04_01.dat"
    db.export_file_db(
        db_name=None,
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


def test_plot_table(io_handler):
    logger.debug("Testing plot_table")

    title = "Test plot table"
    table = astropy.io.ascii.read("tests/resources/Transmission_Spectrum_PlexiGlass.dat")

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


def test_get_telescope_patch(io_handler):
    def test_one_site(x, y):
        _test_radius = 15.0
        for tel_type in names.get_list_of_telescope_types():
            patch = visualize.get_telescope_patch(tel_type, x, y, _test_radius * u.m)
            if mpatches.Circle == type(patch):
                assert patch.radius == _test_radius
            else:
                assert isinstance(patch, mpatches.Rectangle)

    test_one_site(0 * u.m, 0 * u.m)
    test_one_site(0 * u.m, 0 * u.m)
    # Test passing other units
    test_one_site(0 * u.m, 0 * u.km)
    test_one_site(0 * u.cm, 0 * u.km)
    with pytest.raises(TypeError):
        test_one_site(0, 0)


def test_plot_array(
    telescope_north_test_file,
    array_layout_north_instance,
    telescope_south_test_file,
    array_layout_south_instance,
):
    def test_one_site(test_file, instance):
        instance._initialize_array_layout(telescope_list_file=test_file)
        fig_out = visualize.plot_array(
            instance.export_telescope_list_table("ground"), rotate_angle=0 * u.deg
        )
        assert isinstance(fig_out, type(plt.figure()))
        plt.close()

    test_one_site(telescope_north_test_file, array_layout_north_instance)
    test_one_site(telescope_south_test_file, array_layout_south_instance)
