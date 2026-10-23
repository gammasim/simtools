#!/usr/bin/python3

import math

import numpy as np
import pytest

pytest.importorskip("obdeect")

# pylint: disable-next=wrong-import-position
import astropy.units as u
from astropy.constants import c as speed_of_light
from astropy.table import QTable

from simtools.ray_tracing import parameter_changes
from simtools.ray_tracing.backends.interface import LightSourceRequest, StarImageRequest
from simtools.ray_tracing.backends.obdeect import (
    ObdeectBackend,
    ObdeectRayTracer,
    _at_plane,
    star_direction,
)


def test_star_direction():
    assert star_direction(20.0, 0.0, 0.0) == pytest.approx((0.0, 0.0, 1.0), abs=1e-15)
    # Camera x points down: a telescope pointing above the star sees it at +x.
    x, y, _ = star_direction(20.0, 1.0, 0.0)
    assert x == pytest.approx(math.sin(math.radians(1.0)))
    assert y == pytest.approx(0.0, abs=1e-15)
    direction = star_direction(20.0, 0.5, 1.0)
    assert math.degrees(math.asin(direction[1])) == pytest.approx(1.0, rel=1e-3)
    assert np.linalg.norm(direction) == pytest.approx(1.0)


# A toy Davies-Cotton telescope with seven circular 1 m mirrors and f = 8 m.
_MIRROR_LIST = """0 0 100.0 0.0 0
125 0 100.0 0.0 0
-125 0 100.0 0.0 0
62.5 108.3 100.0 0.0 0
-62.5 108.3 100.0 0.0 0
62.5 -108.3 100.0 0.0 0
-62.5 -108.3 100.0 0.0 0
"""


def _record(value, unit=None, file=False):
    return {"value": value, "unit": unit, "file": file, "parameter_version": "1.0.0"}


@pytest.fixture
def toy_telescope(tmp_path, mocker):
    """Telescope model with the parameters obdeect reads for a single-mirror-class dish."""
    (tmp_path / "mirrors.dat").write_text(_MIRROR_LIST, encoding="utf-8")
    (tmp_path / "single_mirror.dat").write_text("0 0 100.0 0.0 0\n", encoding="utf-8")
    parameters = {
        "mirror_class": _record(0),
        "focal_length": _record(800.0, "cm"),
        "mirror_offset": _record(-50.0, "cm"),
        "dish_shape_length": _record(800.0, "cm"),
        "parabolic_dish": _record(False),
        "mirror_focal_length": _record(0.0, "cm"),
        "mirror_list": _record("mirrors.dat", file=True),
        "focus_offset": _record([0.0, 20.0, 0.0, 0.0], ["cm", "deg", None, None]),
        "camera_body_diameter": _record(60.0, "cm"),
        "camera_body_shape": _record(2),
        "camera_depth": _record(0.0, "cm"),
        "random_focal_length": _record([20.0, 0.0], ["cm", "cm"]),
        "mirror_align_random_horizontal": _record(
            [0.02, 20.0, 0.0, 0.0], ["deg", "deg", None, None]
        ),
        "telescope_transmission": _record([0.9, 0, 0, 0, 0, 0]),
        "pixel_shape": _record(1),
    }
    telescope = mocker.Mock(
        parameters=parameters, config_file_directory=tmp_path, model_version="9.9.9"
    )
    telescope.name = "TOY-01"
    telescope.get_parameter_value.side_effect = lambda name: parameters[name]["value"]
    telescope.get_single_mirror_list_file.return_value = tmp_path / "single_mirror.dat"
    return telescope


def test_trace_star(toy_telescope):
    seeds = [(3, 4), (5, 6)]  # (instrument seed, simulation seed) of two instances
    tracer = ObdeectRayTracer(toy_telescope)
    toy_telescope.export_model_files.assert_called_once_with(update_if_necessary=True)
    arrivals = tracer.trace_star(20.0, 0.0, 0.0, 10000.0, 10000, seeds)
    assert arrivals["total_photons"] == 20000
    assert not arrivals["has_structure"]
    # Seven 1 m mirrors minus the shadow of the 0.6 m square camera.
    image = arrivals["position_m"]
    area = len(image) / arrivals["total_photons"] * arrivals["total_area_m2"]
    assert area == pytest.approx(7 * math.pi / 4 - 0.36, rel=0.05)
    assert np.allclose(image[:, 2], 8.5)

    repeated = ObdeectRayTracer(toy_telescope).trace_star(20.0, 0.0, 0.0, 10000.0, 10000, seeds)
    assert np.array_equal(arrivals["position_m"], repeated["position_m"], equal_nan=True)
    first, second = (tracer.optical_model(20.0, seed) for seed in (3, 5))
    assert first is tracer.optical_model(20.0, 3)
    assert first.sha256 != second.sha256  # Independent mirror-error realisations.
    assert first.document["provenance"]["telescope"] == "TOY-01"


def test_perfect_and_single_mirror(toy_telescope):
    tracer = ObdeectRayTracer(toy_telescope)
    perfect = parameter_changes.PERFECT_MIRROR
    normals = [
        [
            f["normal"]
            for f in tracer.optical_model(20.0, s, perfect).document["optics"]["primary"]["facets"]
        ]
        for s in (1, 2)
    ]
    assert normals[0] == normals[1]

    toy_telescope.parameters[parameter_changes.TELESCOPE_STRUCTURE] = _record(
        "structure.ecsv", file=True
    )
    single_mirror = perfect | parameter_changes.single_mirror(
        toy_telescope.get_single_mirror_list_file(0), 800.0, 1600.0
    )
    optics = tracer.optical_model(20.0, 1, single_mirror).document["optics"]
    assert len(optics["primary"]["facets"]) == 1
    assert optics["obscurers"] == []
    assert optics["focal_surface"]["plane_z_m"] == pytest.approx(16.5)
    # The source at the mirror's centre of curvature (distances count from the dish vertex).
    arrivals = tracer.trace_star(20.0, 0.0, 0.0, 16.0, 5000, [(1, 2)], single_mirror)
    image = arrivals["position_m"]
    # Mirror area over the sampling disk (radius 1.2 x 0.5 m): 69%.
    assert len(image) == pytest.approx(0.25 / 0.36 * 5000, rel=0.05)
    # A perfect spherical mirror imaging its centre of curvature: a point image.
    assert np.max(np.hypot(image[:, 0], image[:, 1])) < 1.0e-3


def _request(off_axis_x=0.0):
    return StarImageRequest(
        off_axis_x=off_axis_x,
        off_axis_y=0.0,
        source_distance_km=10.0,
        seeds=((3, 4),),
        photons_per_instance=5000,
    )


@pytest.fixture
def backend(toy_telescope):
    toy_telescope.site = "North"
    return ObdeectBackend(toy_telescope, None, "test", 20.0, False)


@pytest.mark.parametrize("save_photons", [False, True])
def test_backend_psf_image(backend, io_handler, save_photons):
    request = _request()
    with pytest.raises(RuntimeError, match="simulate first"):
        backend.psf_image(request, 800.0, 0.8)
    backend.simulate([request], test=True, force=True, compress_photons=False, max_workers=1)

    image, traced_structure = backend.psf_image(request, 800.0, 0.8, save_photons=save_photons)

    assert traced_structure is False
    assert request.key not in backend._images
    area = image.get_effective_area()
    assert area == pytest.approx(7 * math.pi / 4 - 0.36, rel=0.05)
    files = list(io_handler.get_output_directory().glob("ray_tracing_photons*.ecsv"))
    assert len(files) == int(save_photons)
    if save_photons:
        photons = QTable.read(files[0])
        assert photons["x"].unit == u.cm
        assert photons.meta["total_photons"] == 5000
    with pytest.raises(ValueError, match="rx"):
        backend.psf_image(request, 800.0, 0.8, use_rx=True)


def test_backend_incidence_angles(backend, tmp_path):
    data, files = backend.incidence_angles(_request(1.0), tmp_path, "name", False)
    assert files == []
    assert np.all(np.isfinite(data["angle_incidence_focal"]))
    assert np.median(data["angle_incidence_focal"]) > 1.0  # Off axis and from the dish.


def _light_source(**options):
    return LightSourceRequest(
        **{
            "position_m": (0.1, 0.0, -0.5),
            "axis": (0.0, 0.0, 1.0),
            "angular_profile": ([0.0, 2.0, 2.001], [1.0, 1.0, 0.0]),
            "pulse": ([-1.0, 1.0], [1.0, 1.0]),
            "wavelength_nm": 385.0,
            "photons_per_pulse": (2e5, 2e5),
            "telescope_zenith": 0.0,
            "seeds": (3, 4),
            "inside_telescope": True,
            "camera_radius_m": 0.3,
        }
        | options
    )


def _pulses(backend, request):
    """Photons of each pulse, blocks joined."""
    return [
        {name: np.concatenate([b[name] for b in blocks]) for name in blocks[0]}
        for blocks in (list(pulse) for pulse in backend.camera_photons(request))
    ]


def test_camera_photons_of_a_flasher(backend):
    assert backend.traces_light_sources
    first, second = _pulses(backend, _light_source())
    plane = backend._tracer.optical_model(0.0, 3).focal_surface_z_m
    for photons in (first, second):
        # Straight from the flasher to the camera plane: a 2 deg cone lights 31 cm.
        assert len(photons["weight"]) > 1000
        assert np.allclose(photons["position_m"][:, 2], plane)
        assert np.all(np.hypot(*photons["position_m"][:, :2].T) < 0.3)
        assert np.all((photons["weight"] > 0.0) & (photons["weight"] <= 1.0))
        assert photons["wavelength_nm"] == pytest.approx(385.0)
        assert np.all(photons["direction"][:, 2] > 0.0)
    # Pulses are independent; the same request gives the same photons.
    assert not np.array_equal(first["position_m"][:100], second["position_m"][:100])
    again = _pulses(backend, _light_source())[0]
    assert np.array_equal(again["position_m"], first["position_m"])


def test_camera_photons_of_an_illuminator(backend):
    request = _light_source(
        position_m=(0.0, 0.0, 300.0),
        axis=(0.0, 0.0, -1.0),
        angular_profile=([0.0, 180.0], [1.0, 1.0]),
        pulse=None,
        photons_per_pulse=(1e9,),
        inside_telescope=False,
        camera_radius_m=None,
    )
    (photons,) = _pulses(backend, request)
    # Through the mirrors: the reflectivity weights below one, all at time ~ 300 m / c.
    assert len(photons["weight"]) > 1000
    assert np.all(photons["weight"] < 1.0)
    light_speed = speed_of_light.to_value(u.m / u.ns)
    assert np.median(photons["time_ns"]) == pytest.approx((300.0 + 8.5) / light_speed, rel=0.01)


def test_at_plane():
    photons = {
        "position_m": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        "direction": np.array([[0.0, 0.6, 0.8], [0.0, 0.0, -1.0]]),
        "time_ns": np.array([0.0, 10.0]),
    }
    moved = _at_plane(photons, 0.2, 1.0)
    assert moved["position_m"] == pytest.approx(np.array([[0.0, -0.6, 0.2], [1.0, 0.0, 0.2]]))
    # Backwards along the direction: earlier.
    light_speed = speed_of_light.to_value(u.m / u.ns)
    assert moved["time_ns"] == pytest.approx([-1.0 / light_speed, 10.0 - 0.2 / light_speed])
