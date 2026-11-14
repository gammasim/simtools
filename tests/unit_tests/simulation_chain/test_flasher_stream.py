#!/usr/bin/python3

import numpy as np
import pytest

from simtools.ray_tracing.backends.obdeect import ObdeectBackend, ObdeectRayTracer
from simtools.simulation_chain import flasher_stream
from simtools.simulation_chain.light_sources import (
    AngularDistribution,
    FlasherRequest,
    PulseShape,
    request_to_json,
)

obdeect_stream = pytest.importorskip("obdeect.stream")

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
def toy_telescope(tmp_test_directory, mocker):
    """Telescope model with the parameters obdeect reads for a single-mirror-class dish."""
    (tmp_test_directory / "mirrors.dat").write_text(_MIRROR_LIST, encoding="utf-8")
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
        "mirror_reflection_random_angle": _record([0.0, 0.0, 0.0], ["deg", "deg", None]),
        "mirror_align_random_distance": _record(0.0, "cm"),
        "mirror_align_random_horizontal": _record(
            [0.02, 20.0, 0.0, 0.0], ["deg", "deg", None, None]
        ),
        "mirror_align_random_vertical": _record([0.0, 20.0, 0.0, 0.0], ["deg", "deg", None, None]),
        "telescope_transmission": _record([0.9, 0, 0, 0, 0, 0]),
        "pixel_shape": _record(1),
    }
    telescope = mocker.Mock(
        parameters=parameters, config_file_directory=tmp_test_directory, model_version="9.9.9"
    )
    telescope.name = "TOY-01"
    telescope.site = "North"
    telescope.model_files_exported = False
    telescope.get_parameter_value.side_effect = lambda name: parameters[name]["value"]
    return telescope


def _request(**options):
    return FlasherRequest(
        **{
            "position_m": (0.1, 0.0, -0.5),
            "camera_radius_m": 0.3,
            "events": (1,),
            "photons": (200_000,),
            "wavelength_nm": 385.0,
            "angular": AngularDistribution("FilledCone", 2.0),
            "pulse": PulseShape("Gauss", 0.5, 0.0),
            "observation_level_m": 2158.0,
            "telescope": "TOY-01",
            "site": "North",
            "model_version": "9.9.9",
            "run_number": 3,
            "seed": 4,
            "instrument_seed": 3,
        }
        | options
    )


def _read(path):
    """Return the records of a stream and the concatenated photon columns."""
    records = list(obdeect_stream.Reader(str(path)))
    photons = [r for r in records if r["type"] == "photons"]
    columns = {
        name: np.concatenate([r["columns"][name] for r in photons])
        for name in flasher_stream._STREAM_COLUMNS
    }
    return records, photons, columns


def test_light_source_request():
    light = flasher_stream.light_source_request(_request())
    assert light.position_m == (0.1, 0.0, -0.5)
    assert light.axis == (0.0, 0.0, 1.0)
    assert light.inside_telescope
    assert light.camera_radius_m == pytest.approx(0.3)
    assert light.photons_per_pulse == (200_000,)
    assert light.seeds == (3, 4)
    assert light.telescope_zenith == pytest.approx(0.0)
    assert light.wavelength_nm == pytest.approx(385.0)
    assert light.angular_profile[0][-1] == pytest.approx(2.0)
    assert light.pulse is not None


def test_stream_matches_in_process_tracing(toy_telescope, run_context, tmp_path):
    request = _request()
    output = tmp_path / "flasher.pst"
    total = flasher_stream.write_flasher_stream(ObdeectRayTracer(toy_telescope), request, output)

    backend = ObdeectBackend(toy_telescope, None, "test", 0.0, False, context=run_context)
    (pulse,) = backend.camera_photons(flasher_stream.light_source_request(request))
    blocks = list(pulse)
    expected = {
        name: np.concatenate([block[name] for block in blocks])
        for name in ("position_m", "direction", "time_ns", "weight", "wavelength_nm")
    }
    records, photons, columns = _read(output)

    assert total == len(expected["weight"]) > 1000
    assert [r["type"] for r in records if r["type"] != "photons"] == [
        "stream",
        "run",
        "event",
        "telescope",
        "telescope_end",
        "event_end",
        "run_end",
    ]
    assert all(r["frame"] == 3 for r in photons)
    assert np.allclose(columns["x"], expected["position_m"][:, 0], atol=1e-6)
    assert np.allclose(columns["y"], expected["position_m"][:, 1], atol=1e-6)
    assert np.allclose(columns["cz"], expected["direction"][:, 2], atol=1e-6)
    assert np.allclose(columns["t"], expected["time_ns"], rtol=1e-6, atol=1e-4)
    assert np.allclose(columns["w"], expected["weight"], atol=1e-6)
    assert np.allclose(columns["wavelength"], 385.0)
    # Light of a 2 deg cone from inside the telescope, on the camera: within the camera radius.
    assert np.all(np.hypot(columns["x"], columns["y"]) < 0.3)
    assert np.all(columns["cz"] > 0.0)
    end = next(r for r in records if r["type"] == "telescope_end")
    assert end["metadata"]["photons"] == pytest.approx(float(columns["w"].sum()), rel=1e-6)


def test_stream_chunks(toy_telescope, tmp_path, monkeypatch):
    monkeypatch.setattr(flasher_stream, "MAX_ROWS", 1000)
    output = tmp_path / "flasher.pst"
    total = flasher_stream.write_flasher_stream(ObdeectRayTracer(toy_telescope), _request(), output)
    _, photons, columns = _read(output)
    assert len(columns["x"]) == total
    assert len(photons) > 1
    assert all(len(r["columns"]["x"]) <= 1000 for r in photons)
    assert [r["first_index"] for r in photons[:2]] == [0, len(photons[0]["columns"]["x"])]


def test_main_writes_stream(toy_telescope, tmp_path):
    request_file = request_to_json(_request(), tmp_path / "request.json")
    output = tmp_path / "out.pst"
    seen = {}

    def factory(request, simulation_models_path=None, context=None):
        seen["request"], seen["path"], seen["context"] = request, simulation_models_path, context
        return toy_telescope

    status = flasher_stream.main(
        ["--request", str(request_file), "--output", str(output), "--output-path", str(tmp_path)],
        telescope_model_factory=factory,
    )
    assert status == 0
    assert seen["request"] == _request()
    assert seen["path"] is None
    _, _, columns = _read(output)
    assert len(columns["x"]) > 1000


def test_create_telescope_model_from_files_reader(mocker):
    model = mocker.patch("simtools.simulation_chain.flasher_stream.TelescopeModel")
    reader = mocker.patch("simtools.simulation_chain.flasher_stream.SimulationModelReader")
    flasher_stream.create_telescope_model(_request(), simulation_models_path="models")
    reader.from_files.assert_called_once_with("models")
    model.assert_called_once_with(
        site="North",
        telescope_name="TOY-01",
        model_version="9.9.9",
        model_reader=reader.from_files.return_value,
        context=None,
    )
