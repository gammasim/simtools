"""Tests for the atmosphere comparison application."""

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from simtools.applications import compare_atmospheres

_NORTH = "North"
_PROFILE_VERSION = "1.0.0"
_TRANSMISSION_VERSION = "2.0.0"
_PROFILE_VERSION_KEY = "profile_version"
_TRANSMISSION_VERSION_KEY = "transmission_version"
_RELATIVE_DIFFERENCE_LIMIT = "relative_difference_limit"
_CHARACTERIZE = "characterize_atmospheres"
_READ_ECSV_ASSET = "read_ecsv_asset"


def _start_context(mocker, tmp_test_directory, atmosphere, reader=None):
    context = SimpleNamespace(
        args={"atmosphere": [atmosphere], _RELATIVE_DIFFERENCE_LIMIT: 35.0},
        model_reader=reader,
        io_handler=Mock(),
    )
    context.io_handler.get_output_directory.return_value = tmp_test_directory
    mocker.patch.object(compare_atmospheres.APPLICATION.__class__, "start", return_value=context)
    return context


def test_main_reads_each_parameter_and_runs_comparison(mocker, tmp_test_directory):
    reader = Mock()
    _start_context(
        mocker, tmp_test_directory, [_NORTH, _PROFILE_VERSION, _TRANSMISSION_VERSION], reader
    )
    characterize = mocker.patch.object(compare_atmospheres, _CHARACTERIZE)
    compare_atmospheres.main()
    assert reader.export_model_file.call_count == 2
    inputs = characterize.call_args.args[0]
    assert inputs[0]["site"] == _NORTH
    assert inputs[0][_PROFILE_VERSION_KEY] == _PROFILE_VERSION
    assert inputs[0][_TRANSMISSION_VERSION_KEY] == _TRANSMISSION_VERSION
    assert characterize.call_args.args[1] == tmp_test_directory
    assert math.isclose(characterize.call_args.kwargs[_RELATIVE_DIFFERENCE_LIMIT], 35.0)


def test_main_mixes_ecsv_files_and_versions(mocker, tmp_test_directory):
    profile_file = Path(tmp_test_directory) / "profile.ECSV"
    profile_file.touch()
    reader = Mock()
    _start_context(
        mocker, tmp_test_directory, [_NORTH, str(profile_file), _TRANSMISSION_VERSION], reader
    )
    read_asset = mocker.patch.object(compare_atmospheres, _READ_ECSV_ASSET)
    characterize = mocker.patch.object(compare_atmospheres, _CHARACTERIZE)
    compare_atmospheres.main()
    read_asset.assert_called_once()
    assert read_asset.call_args.args[0] == profile_file
    assert read_asset.call_args.kwargs["schema_entry"]["type"] == "file"
    reader.export_model_file.assert_called_once()
    assert reader.export_model_file.call_args.kwargs["parameter"] == "atmospheric_transmission"
    inputs = characterize.call_args.args[0]
    assert inputs[0]["profile"] is read_asset.return_value
    assert inputs[0][_PROFILE_VERSION_KEY] is None
    assert inputs[0][_TRANSMISSION_VERSION_KEY] == _TRANSMISSION_VERSION
    assert inputs[0]["label"] == f"{_NORTH} reference"


def test_main_labels_reference_and_candidates_per_site(mocker, tmp_test_directory):
    second_north = [_NORTH, _PROFILE_VERSION, _TRANSMISSION_VERSION]
    south = ["South", _PROFILE_VERSION, _TRANSMISSION_VERSION]
    context = _start_context(
        mocker,
        tmp_test_directory,
        [_NORTH, _PROFILE_VERSION, _TRANSMISSION_VERSION],
        Mock(),
    )
    context.args["atmosphere"].extend([second_north, south])
    characterize = mocker.patch.object(compare_atmospheres, _CHARACTERIZE)

    compare_atmospheres.main()

    labels = [item["label"] for item in characterize.call_args.args[0]]
    assert labels == [f"{_NORTH} reference", f"{_NORTH} candidate", "South reference"]


def test_main_with_only_ecsv_files_needs_no_model_reader(mocker, tmp_test_directory):
    files = [Path(tmp_test_directory) / name for name in ("profile.ecsv", "transmission.ecsv")]
    for file in files:
        file.touch()
    _start_context(mocker, tmp_test_directory, [_NORTH, *map(str, files)])
    read_asset = mocker.patch.object(compare_atmospheres, _READ_ECSV_ASSET)
    characterize = mocker.patch.object(compare_atmospheres, _CHARACTERIZE)
    compare_atmospheres.main()
    assert read_asset.call_count == 2
    assert characterize.call_args.args[0][0][_TRANSMISSION_VERSION_KEY] is None


def test_main_raises_for_missing_ecsv_file(mocker, tmp_test_directory):
    missing = Path(tmp_test_directory) / "missing.ecsv"
    _start_context(mocker, tmp_test_directory, [_NORTH, _PROFILE_VERSION, str(missing)], Mock())
    with pytest.raises(FileNotFoundError, match="atmospheric_transmission not found"):
        compare_atmospheres.main()
