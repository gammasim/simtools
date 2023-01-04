#!/usr/bin/python3

import logging

import astropy.io.ascii
import astropy.units as u
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

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


def test_get_telescope_patch(corsika_telescope_data_dict):

    for tel_type in corsika_telescope_data_dict["corsika_sphere_radius"]:
        radius = corsika_telescope_data_dict["corsika_sphere_radius"][tel_type].value
        patch = visualize.get_telescope_patch(tel_type, 0 * u.m, 0 * u.m, radius * u.m)
        if mpatches.Circle == type(patch):
            assert (
                patch.radius == corsika_telescope_data_dict["corsika_sphere_radius"][tel_type].value
            )

        else:
            assert isinstance(patch, mpatches.Rectangle)


def test_plot_array(telescope_test_file, layout_array_north_instance):
    telescope_table = layout_array_north_instance.read_telescope_list_file(telescope_test_file)
    telescopes_dict = layout_array_north_instance.include_radius_into_telescope_table(
        telescope_table
    )
    fig_out = visualize.plot_array(telescopes_dict, rotate_angle=0 * u.deg)
    assert isinstance(fig_out, type(plt.figure()))
