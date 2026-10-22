#!/usr/bin/python3

import math

import numpy as np
import pytest

pytest.importorskip("obdeect")

# pylint: disable-next=wrong-import-position
from simtools.ray_tracing.obdeect_backend import (
    ObdeectRayTracer,
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
    arrivals = tracer.trace_star(20.0, 0.0, 0.0, 10000.0, 20000, seeds)
    assert arrivals["total_photons"] == 20000
    assert not arrivals["has_structure"]
    # Seven 1 m mirrors minus the shadow of the 0.6 m square camera.
    image = arrivals["position_m"]
    area = len(image) / arrivals["total_photons"] * arrivals["total_area_m2"]
    assert area == pytest.approx(7 * math.pi / 4 - 0.36, rel=0.05)
    assert np.allclose(image[:, 2], 8.5)

    repeated = ObdeectRayTracer(toy_telescope).trace_star(20.0, 0.0, 0.0, 10000.0, 20000, seeds)
    assert np.array_equal(arrivals["position_m"], repeated["position_m"], equal_nan=True)
    with pytest.raises(ValueError, match="number_of_photons"):
        tracer.trace_star(20.0, 0.0, 0.0, 10000.0, 1, seeds)
    first, second = (tracer.optical_model(20.0, seed) for seed in (3, 5))
    assert first is tracer.optical_model(20.0, 3)
    assert first.sha256 != second.sha256  # Independent mirror-error realisations.
    assert first.document["provenance"]["telescope"] == "TOY-01"


def test_perfect_and_single_mirror(toy_telescope):
    tracer = ObdeectRayTracer(toy_telescope, perfect_mirror=True)
    normals = [
        [f["normal"] for f in tracer.optical_model(20.0, s).document["optics"]["primary"]["facets"]]
        for s in (1, 2)
    ]
    assert normals[0] == normals[1]

    toy_telescope.parameters["mirror_focal_length"]["value"] = 800.0
    single_mirror = (0, toy_telescope.get_single_mirror_list_file(0), 1600.0)
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
