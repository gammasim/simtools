#!/usr/bin/python3

import dataclasses
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from simtools.corsika.photon_bunches import read_photon_bunches
from simtools.simtel.photon_bunch_file import BUNCH_DTYPE, write_photon_bunches
from simtools.simtel.shower_photons import (
    CORSIKA_LENGTH_UNIT_M,
    CORSIKA_PLANE_Z,
    SPEED_OF_LIGHT_CM_PER_NS,
    ShowerBunchFrame,
    project_to_input_plane,
    shower_bunch_frame,
    write_camera_bunch_file,
)
from simtools.utils import geometry

ZENITH_DIRECTION = np.array([0.0, 0.0, 1.0])
TELESCOPE_AXIS_IN_TELESCOPE_FRAME = np.array([0.0, 0.0, 1.0])
DOWNWARD_IN_TELESCOPE_FRAME = np.array([0.0, 0.0, -1.0])
ATOL = 1e-12
THETA_DEG = 20.0
PHI_DEG = 35.0
SEED = 4242
FAKE_POSITION_M = np.array([0.1, 0.2, 16.0])
FAKE_DIRECTION = np.array([0.0, 0.0, -1.0])
INPUT_ZEM_CM = (216350.0, 200000.0)
INPUT_PLANE_HEIGHT_M = 20.0


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


def test_shower_bunch_frame_fields():
    frame = shower_bunch_frame(theta=20.0, phi=10.0, input_plane_height_m=20.0)

    assert isinstance(frame, ShowerBunchFrame)
    assert {field.name for field in dataclasses.fields(frame)} == {
        "rotation",
        "offset_m",
        "length_unit_m",
        "plane_z",
    }
    assert frame.length_unit_m == pytest.approx(0.01, rel=ATOL)
    assert frame.length_unit_m == pytest.approx(CORSIKA_LENGTH_UNIT_M, rel=ATOL)
    # Input plane height in input units (cm): 100 * 20 m.
    assert frame.plane_z == pytest.approx(2000.0, abs=ATOL)
    assert frame.offset_m.shape == (3,)
    assert frame.offset_m.dtype == np.float64
    np.testing.assert_array_equal(frame.offset_m, np.zeros(3))


class _FakeOpticalModel:
    """Optical model returning one fixed camera photon per input bunch, recording its calls."""

    def __init__(self):
        self.calls = []

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


class _FakeTracer:
    """Tracer returning a fake optical model and recording the optical-model requests."""

    def __init__(self):
        self.model = _FakeOpticalModel()
        self.requests = []

    def optical_model(self, telescope_zenith, instrument_seed, changes=None):
        self.requests.append((telescope_zenith, instrument_seed, changes))
        return self.model


@pytest.fixture
def atmospheric_profile():
    return Table(
        {
            "altitude": [0.0, 2.0, 10.0] * u.km,
            "density": [1.2e-3, 1.0e-3, 4.0e-4] * u.g / u.cm**3,
            "thickness": [1030.0, 800.0, 270.0] * u.g / u.cm**2,
            "refractive_index": [2.8e-4, 2.3e-4, 1.0e-4],
        }
    )


@pytest.fixture
def telescope_model():
    return SimpleNamespace(name="LSTN-01")


def _iact_events(n_bunches, zem_cm):
    bunches = np.zeros(n_bunches, dtype=BUNCH_DTYPE)
    bunches["x"] = np.arange(n_bunches) * 10.0
    bunches["y"] = -np.arange(n_bunches) * 10.0
    bunches["cx"], bunches["cy"] = 0.01, -0.02
    bunches["time"] = 55.0
    bunches["zem"] = zem_cm
    bunches["photons"] = 0.5
    bunches["wavelength"] = 385.0
    return bunches


def _write_iact_file(path, atmospheric_profile):
    events = [
        _iact_events(3, INPUT_ZEM_CM[0]),
        _iact_events(0, INPUT_ZEM_CM[0]),
        _iact_events(2, INPUT_ZEM_CM[1]),
    ]
    write_photon_bunches(
        path,
        events,
        observation_level_m=2147.0,
        telescope_radius_m=2.0,
        atmospheric_profile=atmospheric_profile,
    )
    return path


def _write_camera_file(tracer, iact_file, telescope_model, output_path, atmospheric_profile, **kw):
    return write_camera_bunch_file(
        tracer,
        iact_file,
        telescope_model,
        output_path,
        theta_deg=THETA_DEG,
        phi_deg=PHI_DEG,
        dual_mirror=kw.get("dual_mirror", False),
        camera_rotation_deg=kw.get("camera_rotation_deg", 0.0),
        observation_level_m=2147.0,
        telescope_radius_m=2.0,
        atmospheric_profile=atmospheric_profile,
        seed=SEED,
        telescope_id=0,
        input_plane_height_m=INPUT_PLANE_HEIGHT_M,
        threads=kw.get("threads", 2),
    )


def test_write_camera_bunch_file_passes_tracing_arguments(
    tmp_test_directory, atmospheric_profile, telescope_model
):
    iact_file = _write_iact_file(tmp_test_directory / "input.iact", atmospheric_profile)
    tracer = _FakeTracer()
    output_path = tmp_test_directory / "camera.iact"

    count = _write_camera_file(
        tracer, iact_file, telescope_model, output_path, atmospheric_profile, threads=3
    )

    assert count == 3
    first_event = next(read_photon_bunches(iact_file))
    limits = (first_event["wavelength_min_nm"], first_event["wavelength_max_nm"])
    assert tracer.requests == [(THETA_DEG, SEED, None)]
    # The empty event is not traced.
    assert len(tracer.model.calls) == 2
    frame = shower_bunch_frame(THETA_DEG, PHI_DEG, input_plane_height_m=INPUT_PLANE_HEIGHT_M)
    for bunches, kwargs in tracer.model.calls:
        assert "bunch_id" in bunches
        assert kwargs["seed"] == SEED
        assert kwargs["roulette"] is True
        assert kwargs["normalise_transmission"] is False
        assert kwargs.get("acceptance") is None
        assert kwargs["threads"] == 3
        assert kwargs["wavelength_law"] == limits
        np.testing.assert_array_equal(kwargs["frame_rotation"], frame.rotation)
        np.testing.assert_array_equal(kwargs["frame_offset_m"], frame.offset_m)
        assert kwargs["length_unit_m"] == pytest.approx(CORSIKA_LENGTH_UNIT_M)
        assert kwargs["plane_z"] == pytest.approx(100.0 * INPUT_PLANE_HEIGHT_M)


def test_write_camera_bunch_file_output_readback(
    tmp_test_directory, atmospheric_profile, telescope_model
):
    iact_file = _write_iact_file(tmp_test_directory / "input.iact", atmospheric_profile)
    output_path = tmp_test_directory / "camera.iact"

    _write_camera_file(_FakeTracer(), iact_file, telescope_model, output_path, atmospheric_profile)

    events = list(read_photon_bunches(output_path))
    assert [len(event["x"]) for event in events] == [3, 0, 2]
    assert [event["telescope_id"] for event in events] == [0, 0, 0]
    # Emission altitude [cm]: the input zem of each bunch, unchanged.
    np.testing.assert_allclose(events[0]["zem"], INPUT_ZEM_CM[0], rtol=1e-6)
    np.testing.assert_allclose(events[2]["zem"], INPUT_ZEM_CM[1], rtol=1e-6)
    # Single-mirror camera frame: (x, y) = 100 * (0.1, 0.2) m with y turned.
    np.testing.assert_allclose(events[0]["x"], 10.0, rtol=1e-6)
    np.testing.assert_allclose(events[0]["y"], -20.0, rtol=1e-6)
    np.testing.assert_allclose(events[0]["wavelength_nm"], 385.0, rtol=1e-6)


def test_write_camera_bunch_file_event_count_matches_input(
    tmp_test_directory, atmospheric_profile, telescope_model
):
    iact_file = _write_iact_file(tmp_test_directory / "input.iact", atmospheric_profile)
    output_path = tmp_test_directory / "camera.iact"

    count = _write_camera_file(
        _FakeTracer(), iact_file, telescope_model, output_path, atmospheric_profile
    )

    assert count == len(list(read_photon_bunches(iact_file)))
    assert count == len(list(read_photon_bunches(output_path)))


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
