#!/usr/bin/python3

import math

import numpy as np
import pytest

from simtools.simtel.shower_photons import (
    CORSIKA_PLANE_Z,
    SPEED_OF_LIGHT_CM_PER_NS,
    minimum_input_plane_height_m,
    project_to_input_plane,
    shower_bunch_frame,
)
from simtools.utils import geometry

ZENITH_DIRECTION = np.array([0.0, 0.0, 1.0])
TELESCOPE_AXIS_IN_TELESCOPE_FRAME = np.array([0.0, 0.0, 1.0])
DOWNWARD_IN_TELESCOPE_FRAME = np.array([0.0, 0.0, -1.0])
ATOL = 1e-12
THETA_DEG = 20.0
FAKE_POSITION_M = np.array([0.1, 0.2, 16.0])
FAKE_DIRECTION = np.array([0.0, 0.0, -1.0])
INCOMING_OBSCURER_TOP_M = 27.1


def _assert_proper_rotation(rotation):
    assert rotation.shape == (3, 3)
    assert rotation.dtype == np.float64
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=ATOL)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=ATOL)


@pytest.mark.parametrize("phi", [0.0, 40.0, -120.0])
def test_zenith_pointing_maps_z_to_optical_axis(phi):
    frame = shower_bunch_frame(theta=0.0, phi=phi, input_plane_height_m=20.0)
    _assert_proper_rotation(frame.rotation)

    expected_axis = geometry.ground_direction_to_telescope_frame(ZENITH_DIRECTION, 0.0, phi)
    np.testing.assert_allclose(expected_axis, TELESCOPE_AXIS_IN_TELESCOPE_FRAME, atol=ATOL)
    np.testing.assert_allclose(frame.rotation @ ZENITH_DIRECTION, expected_axis, atol=ATOL)
    np.testing.assert_allclose(frame.rotation[:, 2], expected_axis, atol=ATOL)


@pytest.mark.parametrize(("theta", "phi"), [(30.0, 50.0), (15.0, -70.0), (60.0, 200.0)])
def test_inclined_pointing_arriving_bunch_travels_down_the_axis(theta, phi):
    frame = shower_bunch_frame(theta=theta, phi=phi, input_plane_height_m=20.0)
    _assert_proper_rotation(frame.rotation)

    # Telescope axis in the ground frame: the input vector that maps to +z.
    axis_ground = np.linalg.solve(frame.rotation, TELESCOPE_AXIS_IN_TELESCOPE_FRAME)
    np.testing.assert_allclose(
        geometry.ground_direction_to_telescope_frame(axis_ground, theta, phi),
        TELESCOPE_AXIS_IN_TELESCOPE_FRAME,
        atol=ATOL,
    )
    # The axis is inclined by theta from the ground zenith.
    assert axis_ground @ ZENITH_DIRECTION == pytest.approx(np.cos(np.radians(theta)), abs=ATOL)

    # A bunch arriving along the axis travels towards the ground, i.e. -axis.
    # Expected telescope-frame direction: (0, 0, -1), since photons travel towards -z.
    photon_direction_ground = -axis_ground
    np.testing.assert_allclose(
        frame.rotation @ photon_direction_ground, DOWNWARD_IN_TELESCOPE_FRAME, atol=ATOL
    )
    np.testing.assert_allclose(
        geometry.ground_direction_to_telescope_frame(photon_direction_ground, theta, phi),
        DOWNWARD_IN_TELESCOPE_FRAME,
        atol=ATOL,
    )


class _FakeOpticalModel:
    """Optical model returning one fixed camera photon per input bunch, recording its calls."""

    def __init__(self, top_m=INCOMING_OBSCURER_TOP_M):
        self.calls = []
        self.top_m = top_m

    def max_incoming_obscurer_z_m(self):
        return self.top_m

    def trace_bunches(self, bunches, **kwargs):
        self.calls.append((dict(bunches), kwargs))
        n_bunches = len(bunches["x"])
        return {
            "position_m": np.tile(FAKE_POSITION_M, (n_bunches, 1)),
            "direction": np.tile(FAKE_DIRECTION, (n_bunches, 1)),
            "time_ns": bunches["time_ns"].astype(np.float64),
            "weight": np.ones(n_bunches),
            "wavelength_nm": bunches["wavelength_nm"].astype(np.float64),
            "bunch_index": np.arange(n_bunches, dtype=np.int64),
            "sub_photon": np.zeros(n_bunches, dtype=np.int64),
        }


def _projected_direction(cx, cy):
    cz = -np.sqrt(1.0 - cx**2 - cy**2)
    return np.array([cx, cy, cz])


def test_project_to_input_plane_lies_on_line_through_original_point():
    cx, cy = 0.1, -0.05
    cz = -np.sqrt(1.0 - cx**2 - cy**2)
    origin = np.array([12.0, -7.0, CORSIKA_PLANE_Z])
    bunches = {
        "x": np.array([origin[0]]),
        "y": np.array([origin[1]]),
        "cx": np.array([cx]),
        "cy": np.array([cy]),
        "time_ns": np.array([55.0]),
    }
    height = 2000.0
    projected = project_to_input_plane(bunches, height)
    point = np.array([projected["x"][0], projected["y"][0], height])

    # The projected point lies on the straight line through the original point with the
    # bunch direction (cx, cy, cz): the difference is parallel to that direction.
    difference = point - origin
    direction = _projected_direction(cx, cy)
    np.testing.assert_allclose(direction, [cx, cy, cz], atol=ATOL)
    # The point at height H is reached from the origin by the signed path length H / cz (< 0).
    np.testing.assert_allclose(difference, (height / cz) * direction, atol=1e-9)
    assert np.linalg.norm(difference) == pytest.approx(abs(height / cz), rel=1e-12)
    # Collinear with the bunch direction.
    np.testing.assert_allclose(np.cross(difference, direction), 0.0, atol=1e-9)


def test_project_to_input_plane_time_correction_vertical_bunch():
    height = 2000.0
    bunches = {
        "x": np.array([3.0]),
        "y": np.array([-4.0]),
        "cx": np.array([0.0]),
        "cy": np.array([0.0]),
        "time_ns": np.array([100.0]),
    }
    projected = project_to_input_plane(bunches, height)
    # Vertical bunch at z = 0 with cz = -1: moves to (x, y) at height H, arriving earlier by H / c.
    np.testing.assert_allclose(projected["x"], 3.0, atol=ATOL)
    np.testing.assert_allclose(projected["y"], -4.0, atol=ATOL)
    assert projected["time_ns"][0] == pytest.approx(100.0 - height / SPEED_OF_LIGHT_CM_PER_NS)
    assert projected["time_ns"][0] == pytest.approx(100.0 - 2000.0 / 29.9792458)
    assert projected["x"].dtype == np.float64
    assert projected["time_ns"].dtype == np.float64


def test_project_to_input_plane_inclined_time_correction():
    cx, cy = 0.1, -0.05
    cz = -np.sqrt(1.0 - cx**2 - cy**2)
    bunches = {
        "x": np.array([0.0]),
        "y": np.array([0.0]),
        "cx": np.array([cx]),
        "cy": np.array([cy]),
        "time_ns": np.array([0.0]),
    }
    height = 2000.0
    projected = project_to_input_plane(bunches, height)
    path_length = abs(height / cz)
    assert projected["time_ns"][0] == pytest.approx(-path_length / SPEED_OF_LIGHT_CM_PER_NS)
    assert projected["x"][0] == pytest.approx(height / cz * cx)
    assert projected["y"][0] == pytest.approx(height / cz * cy)


def test_minimum_input_plane_height_m_is_obscurer_top_plus_margin_along_the_axis():
    model = _FakeOpticalModel(top_m=27.1)
    expected = (27.1 + 1.0) / math.cos(math.radians(THETA_DEG))
    assert minimum_input_plane_height_m(model, THETA_DEG) == pytest.approx(expected)
    # The minimum height along the telescope axis is the obscurer top plus the margin.
    assert minimum_input_plane_height_m(model, THETA_DEG) * math.cos(math.radians(THETA_DEG)) == (
        pytest.approx(28.1)
    )
    assert minimum_input_plane_height_m(model, 0.0, margin_m=2.0) == pytest.approx(29.1)


def test_minimum_input_plane_height_m_without_incoming_obscurers():
    model = _FakeOpticalModel(top_m=None)
    assert minimum_input_plane_height_m(model, 0.0) == pytest.approx(1.0)
