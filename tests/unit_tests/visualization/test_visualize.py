#!/usr/bin/python3

import logging

import astropy.io.ascii
import astropy.units as u
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.coordinates.errors import UnitsError

import simtools.util.general as gen
from simtools.layout.layout_array_builder import LayoutArrayBuilder
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
        dest=io_handler.get_output_directory(dir_type="../model", test=True),
        file_name=test_file_name,
    )
    test_data_file = gen.find_file(
        test_file_name, io_handler.get_output_directory(dir_type="../model", test=True)
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
        dest=io_handler.get_output_directory(dir_type="../model", test=True),
        file_name=test_file_name,
    )
    table_file = gen.find_file(
        test_file_name, io_handler.get_output_directory(dir_type="../model", test=True)
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


def test_get_telescope_patch(corsika_telescope_data_dict):

    for tel_type in np.array(list(corsika_telescope_data_dict["corsika_sphere_radius"].keys())):
        radius = corsika_telescope_data_dict["corsika_sphere_radius"][tel_type].value
        patch = visualize.get_telescope_patch(tel_type, 0 * u.m, 0 * u.m, radius * u.m)
        if mpatches.Circle == type(patch):
            assert (
                patch.radius == corsika_telescope_data_dict["corsika_sphere_radius"][tel_type].value
            )

        else:
            assert isinstance(patch, mpatches.Rectangle)


def test_rotate_telescope_position():
    x = np.array([-10.0, -10.0, 10.0, 10.0])
    y = np.array([-10.0, 10.0, -10.0, 10.0])
    angle_deg = 30 * u.deg
    x_rot_manual = np.array([-13.7, -3.7, 3.7, 13.7])
    y_rot_manual = np.array([-3.7, 13.7, -13.7, 3.7])

    def check_results(x_to_test, y_to_test, x_right, y_right):
        x_rot, y_rot = gen.rotate(angle_deg, x_to_test, y_to_test)
        x_rot, y_rot = np.around(x_rot, 1), np.around(y_rot, 1)
        for element, _ in enumerate(x):
            assert x_right[element] == x_rot[element]
            assert y_right[element] == y_rot[element]

    check_results(x, y, x_rot_manual, y_rot_manual)

    x_new_array, y_new_array = x * u.m, y * u.m
    x_rot_new_array, y_rot_new_array = x_rot_manual * u.m, y_rot_manual * u.m
    check_results(x_new_array, y_new_array, x_rot_new_array, y_rot_new_array)

    with pytest.raises(TypeError):
        gen.rotate(angle_deg, x, y[0])
    with pytest.raises(RuntimeError):
        gen.rotate(angle_deg, x[:-1], y)
    with pytest.raises(UnitsError):
        gen.rotate(angle_deg, x_new_array.to(u.cm), y_new_array)


def test_plot_array(telescope_test_file):
    layout_builder_instance = LayoutArrayBuilder()
    telescopes_dict = layout_builder_instance.telescope_layout_file_to_dict(telescope_test_file)
    fig_out = visualize.plot_array(telescopes_dict, rotate_angle=0 * u.deg)
    fig_out.savefig("test.png")
    assert isinstance(fig_out, type(plt.figure()))
