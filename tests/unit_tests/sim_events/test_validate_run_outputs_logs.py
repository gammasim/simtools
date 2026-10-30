#!/usr/bin/python3

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simtools.sim_events.validate_run_outputs import (
    _read_sim_telarray_items,
    assert_events_of_type,
    assert_expected_sim_telarray_metadata,
    assert_expected_sim_telarray_output_and_event_type,
    assert_n_showers_and_energy_range,
    assert_sim_telarray_metadata,
    validate_sim_telarray,
    validate_sim_telarray_event_numbers,
    validate_sim_telarray_logs,
    validate_sim_telarray_metadata,
)


def test_telescope_count_mismatch(tmp_path):
    sim_file = tmp_path / "test.simtel.zst"
    sim_file.write_bytes(b"dummy")

    array_model_mock = MagicMock()
    array_model_mock.site_model.parameters = {}
    array_model_mock.sim_telarray_seed = None
    array_model_mock.telescope_models = {
        "LST-01": MagicMock(),
        "MST-01": MagicMock(),
    }

    with patch(
        "simtools.sim_events.validate_run_outputs.read_sim_telarray_metadata"
    ) as read_meta_mock:
        read_meta_mock.return_value = ({}, {1: {}, 2: {}, 3: {}})

        with pytest.raises(ValueError, match="Number of telescopes"):
            assert_sim_telarray_metadata(sim_file, array_model_mock)


def test_telescope_not_found(tmp_path):
    sim_file = tmp_path / "test.simtel.zst"
    sim_file.write_bytes(b"dummy")

    array_model_mock = MagicMock()
    array_model_mock.site_model.parameters = {}
    array_model_mock.sim_telarray_seed = None
    array_model_mock.telescope_models = {"LST-01": MagicMock()}
    array_model_mock.telescope_models["LST-01"].parameters = {}

    with patch(
        "simtools.sim_events.validate_run_outputs.read_sim_telarray_metadata"
    ) as read_meta_mock:
        read_meta_mock.return_value = ({}, {1: {}})

        with patch(
            "simtools.sim_events.validate_run_outputs.get_sim_telarray_telescope_id"
        ) as get_id_mock:
            get_id_mock.return_value = None

            with pytest.raises(ValueError, match="not found"):
                assert_sim_telarray_metadata(sim_file, array_model_mock)


def test_valid_metadata(tmp_path):
    sim_file = tmp_path / "test.simtel.zst"
    sim_file.write_bytes(b"dummy")

    array_model_mock = MagicMock()
    array_model_mock.site_model.parameters = {}
    array_model_mock.sim_telarray_seed = None
    array_model_mock.telescope_models = {"LST-01": MagicMock()}
    array_model_mock.telescope_models["LST-01"].parameters = {}

    with patch(
        "simtools.sim_events.validate_run_outputs.read_sim_telarray_metadata"
    ) as read_meta_mock:
        read_meta_mock.return_value = ({}, {1: {}})

        with patch(
            "simtools.sim_events.validate_run_outputs.get_sim_telarray_telescope_id"
        ) as get_id_mock:
            get_id_mock.return_value = 1

            assert_sim_telarray_metadata(sim_file, array_model_mock)


@pytest.mark.parametrize(
    ("mock_return", "expected_mc", "expected_shower", "should_raise"),
    [
        ((100, 1000), 1000, 100, False),
        ((50, 500), 1000, 100, True),
    ],
)
def test_validate_sim_telarray_event_numbers(
    tmp_path, mock_return, expected_mc, expected_shower, should_raise
):
    data_file = tmp_path / "test.simtel.zst"
    data_file.write_bytes(b"dummy")

    with patch("simtools.sim_events.file_info.get_simulated_events") as mock_events:
        mock_events.return_value = mock_return
        if should_raise:
            with pytest.raises(ValueError, match="Event mismatch"):
                validate_sim_telarray_event_numbers([data_file], expected_mc, expected_shower)
        else:
            validate_sim_telarray_event_numbers([data_file], expected_mc, expected_shower)


def test_valid_log_files(tmp_path):
    log_file = tmp_path / "test.log"
    log_file.write_text("test log content")

    with patch("simtools.sim_events.validate_run_outputs.check_plain_logs") as mock_check:
        mock_check.return_value = True
        assert validate_sim_telarray_logs([log_file]) is None


def test_invalid_log_files(tmp_path):
    log_file = tmp_path / "test.log"
    log_file.write_text("test log content")

    with patch("simtools.sim_events.validate_run_outputs.check_plain_logs") as mock_check:
        mock_check.return_value = False
        with pytest.raises(ValueError, match="validation failed"):
            validate_sim_telarray_logs([log_file])


def test_valid_showers_and_energy(tmp_path):
    sim_file = tmp_path / "test.simtel.zst"
    sim_file.write_bytes(b"dummy")

    mock_event = {
        "mc_shower": {"energy": 1.0},
        "type": "data",
    }
    mock_run_header = {
        "n_showers": 100,
        "E_range": [0.5, 10.0],
    }

    with patch("simtools.sim_events.validate_run_outputs.SimTelFile") as mock_file:
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.__exit__.return_value = None
        mock_instance.mc_run_headers = [mock_run_header]
        mock_instance.__iter__.return_value = [mock_event] * 99
        mock_file.return_value = mock_instance

        result = assert_n_showers_and_energy_range(sim_file)
        assert result is True


def test_shower_count_mismatch(tmp_path):
    sim_file = tmp_path / "test.simtel.zst"
    sim_file.write_bytes(b"dummy")

    mock_event = {"mc_shower": {"energy": 1.0}, "type": "data"}
    mock_run_header = {"n_showers": 1000, "E_range": [0.5, 10.0]}

    with patch("simtools.sim_events.validate_run_outputs.SimTelFile") as mock_file:
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.__exit__.return_value = None
        mock_instance.mc_run_headers = [mock_run_header]
        mock_instance.__iter__.return_value = [mock_event] * 50
        mock_file.return_value = mock_instance

        with pytest.raises(ValueError, match="does not match"):
            assert_n_showers_and_energy_range(sim_file)


def test_energy_range_exceeded(tmp_path):
    sim_file = tmp_path / "test.simtel.zst"
    sim_file.write_bytes(b"dummy")

    mock_event = {"mc_shower": {"energy": 20.0}, "type": "data"}
    mock_run_header = {"n_showers": 100, "E_range": [0.5, 10.0]}

    with patch("simtools.sim_events.validate_run_outputs.SimTelFile") as mock_file:
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.__exit__.return_value = None
        mock_instance.mc_run_headers = [mock_run_header]
        mock_instance.__iter__.return_value = [mock_event] * 100
        mock_file.return_value = mock_instance

        with pytest.raises(ValueError, match="Energy range"):
            assert_n_showers_and_energy_range(sim_file)


def test_extract_telescope_events(tmp_path):
    sim_file = tmp_path / "test.simtel.zst"
    sim_file.write_bytes(b"dummy")

    mock_event = {
        "type": "data",
        "trigger_information": {"trigger_times": [1.0, 2.0]},
        "photoelectron_sums": {
            "n_pe": np.array([10, 20, 30]),
            "photons": np.array([100, 200, 300]),
            "photons_atm_qe": np.array([50, 100, 150]),
        },
        "telescope_events": [{"data": "test"}],
    }
    expected_output = {
        "trigger_time": [1.0, 2.0],
        "pe_sum": [10, 20, 30],
        "photons": [50, 100, 150],
    }

    with patch("simtools.sim_events.validate_run_outputs.SimTelFile") as mock_file:
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.__exit__.return_value = None
        mock_instance.__iter__.return_value = [mock_event]
        mock_file.return_value = mock_instance

        result = _read_sim_telarray_items(sim_file, expected_output, set())

        assert "n_telescope_events" in result
        assert result["n_telescope_events"] == 1


def test_expected_output_and_event_type_uses_one_traversal(tmp_path):
    sim_file = tmp_path / "test.simtel.zst"
    sim_file.write_bytes(b"dummy")
    mock_event = {
        "type": "data",
        "trigger_information": {"trigger_times": [1.0]},
        "photoelectron_sums": {
            "n_pe": np.array([10]),
            "photons": np.array([100]),
            "photons_atm_qe": np.array([50]),
        },
        "telescope_events": [{"data": "test"}],
    }

    with patch("simtools.sim_events.validate_run_outputs.SimTelFile") as mock_file:
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.__exit__.return_value = None
        mock_instance.__iter__.return_value = [mock_event]
        mock_file.return_value = mock_instance

        result = assert_expected_sim_telarray_output_and_event_type(
            sim_file,
            {"event_type": "shower", "pe_sum": [0.0, 100.0]},
            event_type="shower",
        )

    assert result is True
    mock_file.assert_called_once_with(sim_file)


def test_none_expected_metadata():
    result = assert_expected_sim_telarray_metadata(Path("/dummy/file.zst"), None)
    assert result is True


@pytest.mark.parametrize(
    ("expected_metadata", "metadata_return", "expected_result"),
    [
        ({"key": "value"}, ({"key": "value"}, {}), True),
        ({"missing_key": "value"}, ({}, {}), False),
        ({"key": "expected_value"}, ({"key": "actual_value"}, {}), False),
    ],
)
def test_assert_expected_sim_telarray_metadata(expected_metadata, metadata_return, expected_result):
    with patch("simtools.sim_events.validate_run_outputs.read_sim_telarray_metadata") as mock_read:
        mock_read.return_value = metadata_return
        result = assert_expected_sim_telarray_metadata(Path("/dummy/file.zst"), expected_metadata)
        assert result is expected_result


@pytest.mark.parametrize(
    ("mock_events", "event_type", "expected_result"),
    [
        ([{"type": "data"}], "shower", True),
        ([{"type": "calibration"}], "pedestal", True),
        ([], "shower", False),
    ],
)
def test_assert_events_of_type(tmp_path, mock_events, event_type, expected_result):
    sim_file = tmp_path / "test.simtel.zst"
    sim_file.write_bytes(b"dummy")

    with patch("simtools.sim_events.validate_run_outputs.SimTelFile") as mock_file:
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.__exit__.return_value = None
        mock_instance.__iter__.return_value = mock_events
        mock_file.return_value = mock_instance

        result = assert_events_of_type(sim_file, event_type)
        assert result is expected_result


@pytest.mark.parametrize(
    ("filename", "model_version", "should_call_assert"),
    [
        ("model_v1.0.0.simtel.zst", "1.0.0", True),
        ("other_v2.0.0.simtel.zst", "1.0.0", False),
    ],
)
def test_validate_sim_telarray_metadata(tmp_path, filename, model_version, should_call_assert):
    test_file = tmp_path / filename
    test_file.write_bytes(b"dummy")

    model_mock = MagicMock()
    model_mock.model_version = model_version

    with patch(
        "simtools.sim_events.validate_run_outputs.assert_sim_telarray_metadata"
    ) as mock_assert:
        validate_sim_telarray_metadata([test_file], [model_mock])
        if should_call_assert:
            mock_assert.assert_called_once()
        else:
            mock_assert.assert_not_called()


@pytest.mark.parametrize(
    ("array_models", "should_call_metadata"),
    [
        (None, False),
        ([MagicMock()], True),
    ],
)
def test_validate_sim_telarray(tmp_path, array_models, should_call_metadata):
    data_file = tmp_path / "test.simtel.zst"
    data_file.write_bytes(b"dummy")
    log_file = tmp_path / "test.log"
    log_file.write_text("test log")

    with patch("simtools.sim_events.validate_run_outputs.validate_sim_telarray_logs") as mock_log:
        with patch(
            "simtools.sim_events.validate_run_outputs.validate_sim_telarray_metadata"
        ) as mock_meta:
            validate_sim_telarray([data_file], [log_file], array_models=array_models)
            mock_log.assert_called_once()
            if should_call_metadata:
                mock_meta.assert_called_once()
            else:
                mock_meta.assert_not_called()


def test_validate_sim_telarray_checks_zero_expected_events(tmp_path):
    data_file = tmp_path / "test.simtel.zst"
    data_file.write_bytes(b"dummy")
    log_file = tmp_path / "test.log"
    log_file.write_text("test log")

    with (
        patch("simtools.sim_events.validate_run_outputs.validate_sim_telarray_logs"),
        patch(
            "simtools.sim_events.validate_run_outputs.validate_sim_telarray_event_numbers"
        ) as mock_events,
    ):
        validate_sim_telarray(
            [data_file], [log_file], expected_mc_events=0, expected_shower_events=0
        )

    mock_events.assert_called_once_with([data_file], 0, 0)


def test_assert_events_of_type_not_found(tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    data_file = tmp_path / "test.simtel.zst"
    data_file.write_bytes(b"dummy")

    with patch("simtools.sim_events.validate_run_outputs.SimTelFile") as mock_file:
        mock_file.return_value.__enter__.return_value = iter([{"type": "calibration"}])
        result = assert_events_of_type(data_file, event_type="shower")
        assert result is False
        assert "No events of type" in caplog.text
