#!/usr/bin/python3

import logging

import astropy.units as u
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pytest

from simtools.layout.layout_array_builder import LayoutArrayBuilder

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def layout_builder_instance(io_handler):
    return LayoutArrayBuilder()


def test_telescope_layout_file_to_dict(telescope_test_file, layout_builder_instance):
    telescopes_dict = layout_builder_instance.telescope_layout_file_to_dict(telescope_test_file)
    values_from_file = [20.190000534057617, -352.4599914550781, 62.29999923706055, 9.6]
    keys = ["pos_x", "pos_y", "pos_z", "radius"]
    MST10_index = telescopes_dict["telescope_name"] == "MST-10"
    for key_step, _ in enumerate(keys):
        assert telescopes_dict[MST10_index][keys[key_step]].value[0] == values_from_file[key_step]


def test_get_telescope_patch(corsika_telescope_data_dict, layout_builder_instance):
    for tel_type in np.array(list(corsika_telescope_data_dict["corsika_sphere_radius"].keys())):
        radius = corsika_telescope_data_dict["corsika_sphere_radius"][tel_type]
        patch = layout_builder_instance._get_telescope_patch(
            tel_type, 0 * u.m, 0 * u.m, radius.value * u.m
        )
        assert mpatches.Circle == type(patch)


def test_rotate_telescope_position(layout_builder_instance):
    x = np.array([-1, -1, 1, 1])
    y = np.array([-1, 1, -1, 1])
    angle_deg = 30
    x_rot_manual = np.array([-1.37, -0.37, 0.37, 1.37])
    y_rot_manual = np.array([-0.37, 1.37, -1.37, 0.37])
    x_rot, y_rot = layout_builder_instance._rotate(angle_deg, x, y)
    x_rot, y_rot = np.around(x_rot, 2), np.around(y_rot, 2)
    for element in range(len(x)):
        assert x_rot_manual[element] == x_rot[element]
        assert y_rot_manual[element] == y_rot[element]


def test_plot_array(telescope_test_file, layout_builder_instance):
    telescopes_dict = layout_builder_instance.telescope_layout_file_to_dict(telescope_test_file)
    fig_out = layout_builder_instance.plot_array(telescopes_dict, rotate_angle=0)
    assert isinstance(fig_out, type(plt.figure()))
