#!/usr/bin/python3

import pytest

from simtools.ray_tracing import parameter_changes


def test_single_mirror():
    changes = parameter_changes.single_mirror("mirror.dat", 1600.0, 3200.0)
    assert changes["mirror_list"] == "mirror.dat"
    assert changes["focal_length"] == pytest.approx(3200.0)
    assert changes["dish_shape_length"] == pytest.approx(1600.0)
    assert changes["mirror_focal_length"] == pytest.approx(1600.0)
    assert changes["parabolic_dish"] is False


def test_sim_telarray_options():
    options = parameter_changes.sim_telarray_options(
        {"parabolic_dish": False, "focus_offset": [0.0, 28.0], "focal_length": 3200.0}
    )
    assert options == {"parabolic_dish": "0", "focus_offset": "0.0,28.0", "focal_length": "3200.0"}
    assert set(parameter_changes.sim_telarray_options(parameter_changes.PERFECT_MIRROR)) == set(
        parameter_changes.PERFECT_MIRROR
    )
