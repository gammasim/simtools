#!/usr/bin/python3

import logging

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
    values_from_file = np.array([20.190000534057617, -352.4599914550781, 62.29999923706055, 9.6])
    keys = ["pos_x", "pos_y", "pos_z", "radius"]
    mst_10_index = telescopes_dict["telescope_name"] == "MST-10"
    for key_step, _ in enumerate(keys):
        assert telescopes_dict[mst_10_index][keys[key_step]].value[0] == values_from_file[key_step]
