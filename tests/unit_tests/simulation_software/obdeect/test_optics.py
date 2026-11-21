"""Tests for the optical model file and the command line of the obdeect ray tracing."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simtools.simulation_software.obdeect import optics
from simtools.simulation_software.optics import OpticsRequest
from simtools.simulation_software.programs import program_command

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
def toy_telescope(tmp_test_directory):
    """Telescope model of a single-mirror dish with seven 1 m mirrors (f = 8 m)."""
    Path(tmp_test_directory / "model" / "mirrors.dat").write_text(_MIRROR_LIST, encoding="utf-8")
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
    return SimpleNamespace(
        name="TOY-01",
        model_version="9.9.9",
        model_files_exported=True,
        parameters=parameters,
        config_file_directory=Path(tmp_test_directory) / "model",
        get_parameter_value=lambda name: parameters[name]["value"],
    )


@pytest.fixture
def request_one():
    return OpticsRequest(
        telescope_type="TOY-01",
        telescope_name="TOYN-02",
        telescope_id=3,
        model_version="9.9.9",
        site="North",
        dual_mirror=False,
        zenith_deg=20.0,
        azimuth_deg=180.0,
        instrument_seed=3,
        tracing_seed=4,
    )


def test_trace_stream_command(request_one):
    other = replace(request_one, telescope_name="TOYN-05", telescope_id=5)

    command = optics.trace_stream_command(
        [request_one, other], ["toy2.json", "toy5.json"], "in.pst", "out.pst"
    )

    assert command == [
        *program_command("obdeect", "obdeect"),
        "ray-tracing",
        "trace-stream",
        "--model",
        "3=toy2.json",
        "--model",
        "5=toy5.json",
        "--seed",
        "4",
        "--threads",
        "1",
        "--pointing-zenith-deg",
        "20.0",
        "--pointing-azimuth-deg",
        "180.0",
        "--telescope-transmission",
        "full",
        "--input",
        "in.pst",
        "--output",
        "out.pst",
    ]


def test_trace_stream_command_streams_by_default(request_one):
    command = optics.trace_stream_command([request_one], ["model.json"])

    assert command[command.index("--input") + 1] == "-"
    assert command[command.index("--output") + 1] == "-"


@pytest.mark.parametrize(
    ("requests", "files", "match"),
    [
        ([], [], "one model file per request; got 0 requests"),
        (["one"], ["a.json", "b.json"], "got 1 requests and 2 model files"),
        (["one", "pointing"], ["a.json", "b.json"], "differ in zenith_deg"),
        (["one", "seed"], ["a.json", "b.json"], "differ in tracing_seed"),
    ],
)
def test_trace_stream_command_rejects_inconsistent_requests(request_one, requests, files, match):
    variants = {
        "one": request_one,
        "pointing": replace(request_one, telescope_id=5, zenith_deg=30.0),
        "seed": replace(request_one, telescope_id=5, tracing_seed=9),
    }

    with pytest.raises(ValueError, match=match):
        optics.trace_stream_command([variants[name] for name in requests], files)


def test_write_optical_model(request_one, toy_telescope):
    obdeect = pytest.importorskip("obdeect.ray_tracing")

    path = optics.write_optical_model(request_one, toy_telescope)

    assert path.parent == toy_telescope.config_file_directory
    assert path.name == request_one.model_file_name
    model = obdeect.OpticalModel.load(path)
    assert model.document["provenance"]["telescope"] == "TOY-01"


def test_optical_model_depends_on_instrument_seed(request_one, toy_telescope):
    pytest.importorskip("obdeect.ray_tracing")

    first = optics.write_optical_model(request_one, toy_telescope)
    other = optics.write_optical_model(replace(request_one, instrument_seed=5), toy_telescope)

    assert first != other
    assert first.read_text(encoding="utf-8") != other.read_text(encoding="utf-8")


def test_trace_stream_of_ground_photons(request_one, toy_telescope, tmp_test_directory):
    """Run obdeect on a small ground-frame stream: two telescopes in one process, one dropped."""
    stream = pytest.importorskip("obdeect.stream")
    cli = pytest.importorskip("obdeect.stream.cli")
    model_file = optics.write_optical_model(request_one, toy_telescope)
    count = 200
    rng = np.random.default_rng(1)
    zenith = np.radians(request_one.zenith_deg)
    columns = {
        "x": rng.uniform(-5, 5, count).astype("f4"),
        "y": rng.uniform(-5, 5, count).astype("f4"),
        "cx": np.full(count, -np.sin(zenith), "f4"),
        "cy": np.zeros(count, "f4"),
        "t": np.zeros(count, "f4"),
        "w": np.ones(count, "f4"),
        "zem": np.full(count, 20000.0, "f4"),
        "wavelength": np.full(count, 400.0, "f4"),
    }
    input_file = Path(tmp_test_directory) / "ground.pst"
    output_file = Path(tmp_test_directory) / "camera.pst"
    with stream.Writer(str(input_file), "test") as writer:
        writer.write(
            "run",
            {
                "run_number": 1,
                "observation_level_m": 2200.0,
                "n_telescopes": 3,
                "telescope_x_m": [0.0, 10.0, 20.0],
                "telescope_y_m": [0.0, 0.0, 0.0],
                "telescope_z_m": [0.0, 0.0, 0.0],
                "telescope_r_m": [10.0, 10.0, 10.0],
            },
        )
        writer.write("event", {"event_number": 1})
        for telescope_id in (0, 1, 3):
            writer.write(
                "telescope", {"telescope_id": telescope_id, "frame": stream.FRAMES["ground"]}
            )
            writer.write_photons(telescope_id, "ground", 0, columns)
            writer.write("telescope_end", {"telescope_id": telescope_id})
        writer.write("event_end", {"event_number": 1})
        writer.write("run_end", {})

    other_request = replace(request_one, telescope_name="TOYN-01", telescope_id=1)
    command = optics.trace_stream_command(
        [request_one, other_request], [model_file, model_file], input_file, output_file
    )

    assert cli.main(command[command.index("ray-tracing") :]) == 0
    records = list(stream.Reader(str(output_file)))
    telescopes = [
        record["metadata"]["telescope_id"] for record in records if record["type"] == "telescope"
    ]
    assert telescopes == [1, 3]
    assert len([r for r in records if r["type"] == "event"]) == 1
    assert all(
        record["metadata"]["frame"] == stream.FRAMES["camera"]
        for record in records
        if record["type"] == "telescope"
    )
    simtel_file = Path(tmp_test_directory) / "camera.iact"
    roundtrip_file = Path(tmp_test_directory) / "roundtrip.pst"
    # the toy stream has no atmosphere status; sim_telarray would apply the atmosphere
    stream.stream_to_simtel(
        output_file, simtel_file, {1: (1, 0.0), 3: (1, 0.0)}, allow_pending_atmosphere=True
    )
    stream.corsika_to_stream(simtel_file, roundtrip_file)
    roundtrip = list(stream.Reader(str(roundtrip_file)))
    assert {r["metadata"]["telescope_id"] for r in roundtrip if r["type"] == "telescope"} == {1, 3}
