"""Tests for the optics request of the simulation chain and its obdeect command."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simtools.simtel.shower_photons import derive_seed
from simtools.simulation_chain import optics
from simtools.simulation_chain.optics import OpticsRequest

_MIRROR_LIST = """0 0 100.0 0.0 0
125 0 100.0 0.0 0
-125 0 100.0 0.0 0
62.5 108.3 100.0 0.0 0
-62.5 108.3 100.0 0.0 0
62.5 -108.3 100.0 0.0 0
-62.5 -108.3 100.0 0.0 0
"""
_NOT_IN_RANGE = "must lie in"


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


def _array_model(types_by_name, seed_string="11,22"):
    models = {
        name: SimpleNamespace(
            name=type_name,
            get_parameter_value=lambda _name, mirror_class=mirror_class: mirror_class,
        )
        for name, (type_name, mirror_class) in types_by_name.items()
    }
    return SimpleNamespace(
        telescope_models=models,
        model_version="7.0.0",
        site="South",
        sim_telarray_seed=SimpleNamespace(seed_string=seed_string, instrument_seed=None),
    )


def test_seeds_of_seed_string():
    seeds = SimpleNamespace(seed_string="11,22", instrument_seed=None)
    assert optics.seeds_of(seeds) == (11, 22)


def test_seeds_of_seed_file():
    seeds = SimpleNamespace(
        seed_string="file-by-run:seeds.txt", instrument_seed="5", simulation_seed="6"
    )
    assert optics.seeds_of(seeds) == (5, 6)


def test_seeds_of_seed_file_without_instrument_seed():
    seeds = SimpleNamespace(seed_string="file-by-run:seeds.txt", instrument_seed=None)
    with pytest.raises(ValueError, match="Instrument seed is not set"):
        optics.seeds_of(seeds)


def test_request_is_frozen(request_one):
    with pytest.raises(AttributeError):
        request_one.threads = 2


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"telescope_transmission": "none"}, "Invalid telescope transmission"),
        ({"telescope_transmission": "angular"}, "stay unapplied"),
        ({"telescope_id": -1}, "telescope id"),
        ({"zenith_deg": 90.0}, _NOT_IN_RANGE),
        ({"zenith_deg": -1.0}, _NOT_IN_RANGE),
        ({"tracing_seed": -1}, "must not be negative"),
        ({"threads": -1}, "must not be negative"),
    ],
)
def test_request_validation(request_one, change, match):
    with pytest.raises(ValueError, match=match):
        replace(request_one, **change)


def test_optics_requests_one_per_telescope():
    array_model = _array_model(
        {"LSTS-01": ("LST", 1), "MSTS-01": ("MST", 2), "LSTS-02": ("LST", 1)}
    )
    parameters = SimpleNamespace(zenith_angle=20.0, azimuth_angle=180.0)
    ids = {"LSTS-01": 0, "MSTS-01": 1, "LSTS-02": 2}

    requests = optics.optics_requests(array_model, parameters, ids, stream_id_offset=1, threads=2)

    assert [request.telescope_name for request in requests] == ["LSTS-01", "MSTS-01", "LSTS-02"]
    assert [request.telescope_type for request in requests] == ["LST", "MST", "LST"]
    assert [request.telescope_id for request in requests] == [1, 2, 3]
    assert [request.dual_mirror for request in requests] == [False, True, False]
    first = requests[0]
    assert (first.model_version, first.site) == ("7.0.0", "South")
    assert (first.zenith_deg, first.azimuth_deg) == (20.0, 180.0)
    assert (first.tracing_seed, first.telescope_transmission, first.threads) == (22, "full", 2)


def test_instrument_seed_of_each_telescope_is_derived_from_its_iact_id():
    array_model = _array_model({"LSTS-01": ("LST", 1), "LSTS-02": ("LST", 1)})
    parameters = SimpleNamespace(zenith_angle=20.0, azimuth_angle=180.0)
    ids = {"LSTS-01": 0, "LSTS-02": 1}

    first, second = optics.optics_requests(array_model, parameters, ids, stream_id_offset=1)

    assert first.instrument_seed == derive_seed(11, 0)
    assert second.instrument_seed == derive_seed(11, 1)
    assert first.instrument_seed != second.instrument_seed
    assert first.model_file_name != second.model_file_name


def test_trace_stream_command(request_one):
    command = optics.trace_stream_command(request_one, "model.json", "in.pst", "out.pst")

    assert command == [
        "obdeect",
        "ray-tracing",
        "trace-stream",
        "--model",
        "model.json",
        "--seed",
        "4",
        "--threads",
        "1",
        "--telescopes",
        "3",
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
    command = optics.trace_stream_command(request_one, "model.json")

    assert command[command.index("--input") + 1] == "-"
    assert command[command.index("--output") + 1] == "-"


def test_write_telescope_map(request_one, tmp_test_directory):
    path = optics.write_telescope_map([request_one], Path(tmp_test_directory) / "map.json")

    entries = json.loads(path.read_text(encoding="utf-8"))
    assert list(entries) == ["3"]
    assert entries["3"]["name"] == "TOYN-02"
    assert entries["3"]["model"] == "TOY-01"
    assert entries["3"]["phi_deg"] == pytest.approx(180.0)


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
    """Run obdeect on a small ground-frame stream: unselected telescopes are dropped."""
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
        "wavelength": np.full(count, 400.0, "f4"),
    }
    input_file = Path(tmp_test_directory) / "ground.pst"
    output_file = Path(tmp_test_directory) / "camera.pst"
    with stream.Writer(str(input_file), "test") as writer:
        writer.write("run", {"run_number": 1})
        writer.write("event", {"event_number": 1})
        for telescope_id in (0, 1, 3):
            writer.write(
                "telescope", {"telescope_id": telescope_id, "frame": stream.FRAMES["ground"]}
            )
            writer.write_photons(telescope_id, "ground", 0, columns)
            writer.write("telescope_end", {"telescope_id": telescope_id})
        writer.write("event_end", {"event_number": 1})
        writer.write("run_end", {})

    command = optics.trace_stream_command(request_one, model_file, input_file, output_file)

    assert cli.main(command[1:]) == 0
    records = list(stream.Reader(str(output_file)))
    telescopes = [
        record["metadata"]["telescope_id"] for record in records if record["type"] == "telescope"
    ]
    assert telescopes == [3]
    assert all(
        record["metadata"]["frame"] == stream.FRAMES["camera"]
        for record in records
        if record["type"] == "telescope"
    )
