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
    assert path.read_text(encoding="utf-8").splitlines() == ["0 1 0.0", "1 2 30.0"]
