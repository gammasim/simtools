"""Tests for the commands of the obdeect stream adapters."""

from pathlib import Path
from types import SimpleNamespace

import astropy.units as u

from simtools.simulation_software.obdeect import streams
from simtools.simulation_software.programs import program_command


def test_from_corsika_command():
    assert streams.from_corsika_command("run.iact") == [
        *program_command("obdeect", "obdeect-stream-from-corsika"),
        "run.iact",
    ]


def test_to_simtel_command_with_atmprof():
    command = streams.to_simtel_command("map.txt", "in.pst", atmprof="atmprof.dat")

    assert command == [
        *program_command("obdeect", "obdeect-stream-to-simtel"),
        *("--camera-map", "map.txt"),
        *("--atmprof", "atmprof.dat"),
        "in.pst",
    ]


def test_to_simtel_command_with_factors():
    command = streams.to_simtel_command(
        "map.txt", "in.pst", allow_pending=("a", "b"), scaled_factors=("c",)
    )

    assert command == [
        *program_command("obdeect", "obdeect-stream-to-simtel"),
        *("--camera-map", "map.txt"),
        *("--allow-pending", "a", "--allow-pending", "b"),
        *("--scaled-factor", "c"),
        "in.pst",
    ]


def test_write_camera_map(tmp_test_directory, monkeypatch):
    models = {
        name: SimpleNamespace(
            get_parameter_value=lambda _name, value=mirror_class: value,
            get_parameter_value_with_unit=lambda _name, value=rotation: value * u.deg,
        )
        for name, mirror_class, rotation in (("LSTN-01", 1, 0.0), ("MSTN-01", 2, 30.0))
    }
    monkeypatch.setattr(
        streams, "corsika_telescope_ids", lambda _models: {"LSTN-01": 0, "MSTN-01": 1}
    )
    path = Path(tmp_test_directory) / "map.txt"

    assert streams.write_camera_map(models, path) == path
    assert path.read_text(encoding="utf-8").splitlines() == ["1 1 0.0", "2 2 30.0"]


def test_camera_map_ids_are_stream_ids_from_one(tmp_test_directory):
    """Stream ids start at 1 (IACT id 0 is stream 1); stream-to-simtel rejects 0."""
    models = {
        "LSTN-01": SimpleNamespace(
            get_parameter_value=lambda _n: 1, get_parameter_value_with_unit=lambda _n: 0 * u.deg
        )
    }
    path = Path(tmp_test_directory) / "map_ids.txt"

    streams.write_camera_map(models, path)

    assert path.read_text(encoding="utf-8").split()[0] == "1"


def test_camera_map_mirror_class_single_is_one(tmp_test_directory):
    """The model mirror class 0 (single mirror) is written as 1; dual mirror 2 stays 2."""
    models = {
        "LSTN-01": SimpleNamespace(
            get_parameter_value=lambda _n: 0,
            get_parameter_value_with_unit=lambda _n: 0 * u.deg,
        )
    }
    path = Path(tmp_test_directory) / "map_class.txt"

    streams.write_camera_map(models, path)

    assert path.read_text(encoding="utf-8").split()[1] == "1"
