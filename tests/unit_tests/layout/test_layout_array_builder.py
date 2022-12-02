#!/usr/bin/python3

import logging

import astropy.units as u
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.coordinates.errors import UnitsError

from simtools.layout.layout_array_builder import LayoutArrayBuilder
from simtools.util import general as gen
from simtools.visualization import visualize

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def layout_builder_instance(io_handler):
    return LayoutArrayBuilder()


def test_telescope_layout_file_to_dict(telescope_test_file, layout_builder_instance):
    telescopes_dict = layout_builder_instance.telescope_layout_file_to_dict(telescope_test_file)
    values_from_file = np.array([20.190000534057617, -352.4599914550781, 62.29999923706055, 9.6])
    keys = ["pos_x", "pos_y", "pos_z", "radius"]
    mst_10_index = telescopes_dict["telescope_name"] == "MST-10"
    for key_step, _ in enumerate(keys):
        assert telescopes_dict[mst_10_index][keys[key_step]].value[0] == values_from_file[key_step]


def test_get_telescope_patch(corsika_telescope_data_dict, layout_builder_instance):

    for tel_type in np.array(list(corsika_telescope_data_dict["corsika_sphere_radius"].keys())):
        print(corsika_telescope_data_dict)
        radius = corsika_telescope_data_dict["corsika_sphere_radius"][tel_type].value
        patch = visualize.get_telescope_patch(tel_type, 0 * u.m, 0 * u.m, radius * u.m)
        if mpatches.Circle == type(patch):
            assert (
                patch.radius == corsika_telescope_data_dict["corsika_sphere_radius"][tel_type].value
            )

        else:
            assert isinstance(patch, mpatches.Rectangle)


def test_rotate_telescope_position(layout_builder_instance):
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


def test_plot_array(telescope_test_file, layout_builder_instance):
    telescopes_dict = layout_builder_instance.telescope_layout_file_to_dict(telescope_test_file)
    fig_out = layout_builder_instance.plot_array(telescopes_dict, rotate_angle=0)
    assert isinstance(fig_out, type(plt.figure()))
