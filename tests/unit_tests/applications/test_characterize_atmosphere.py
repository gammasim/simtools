"""Tests for the atmospheric characterization application."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

from simtools.applications import characterize_atmosphere

_NORTH = "North"
_PROFILE_VERSION = "1.0.0"
_TRANSMISSION_VERSION = "2.0.0"


def test_main_reads_each_parameter_and_runs_characterization(mocker, tmp_test_directory):
    reader = Mock()
    context = SimpleNamespace(
        args={
            "atmosphere": [[_NORTH, _PROFILE_VERSION, _TRANSMISSION_VERSION]],
            "relative_difference_limit": 35.0,
        },
        model_reader=reader,
        io_handler=Mock(),
    )
    context.io_handler.get_output_directory.return_value = tmp_test_directory
    mocker.patch.object(
        characterize_atmosphere.APPLICATION.__class__, "start", return_value=context
    )
    characterize = mocker.patch.object(characterize_atmosphere, "characterize_atmospheres")
    characterize_atmosphere.main()
    assert reader.export_model_file.call_count == 2
    inputs = characterize.call_args.args[0]
    assert inputs[0]["site"] == _NORTH
    assert inputs[0]["profile_version"] == _PROFILE_VERSION
    assert inputs[0]["transmission_version"] == _TRANSMISSION_VERSION
    assert characterize.call_args.args[1] == tmp_test_directory
    assert math.isclose(characterize.call_args.kwargs["relative_difference_limit"], 35.0)
