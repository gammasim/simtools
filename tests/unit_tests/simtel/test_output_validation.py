#!/usr/bin/python3

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simtools.simtel.output_validation import (
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
from simtools.simulation.nsb import NsbSettings, airmass_scale


def _sim_telarray_writer(settings=None, written_nsb=None, exported=True):
    """Return a sim_telarray configuration writer mock with recorded NSB values."""
    writer = MagicMock()
    writer.telescope_model_files_exported = exported
    writer.array_model_file_exported = exported
    writer.nsb_settings = settings
    writer.written_nsb_parameters = written_nsb
    return writer


def test_telescope_count_mismatch(tmp_path):
    sim_file = tmp_path / "test.simtel.zst"
    sim_file.write_bytes(b"dummy")

    array_model_mock = MagicMock()
    array_model_mock.site_model.parameters = {}
    array_model_mock.sim_telarray_seed = None
    array_model_mock.configuration_writers = {"sim_telarray": _sim_telarray_writer()}
    array_model_mock.telescope_models = {
        "LST-01": MagicMock(),
        "MST-01": MagicMock(),
    }

    with patch("simtools.simtel.output_validation.read_sim_telarray_metadata") as read_meta_mock:
        read_meta_mock.return_value = ({}, {1: {}, 2: {}, 3: {}})

        with pytest.raises(ValueError, match="Number of telescopes"):
            assert_sim_telarray_metadata(sim_file, array_model_mock)


def test_telescope_not_found(tmp_path):
    sim_file = tmp_path / "test.simtel.zst"
    sim_file.write_bytes(b"dummy")

    array_model_mock = MagicMock()
    array_model_mock.site_model.parameters = {}
    array_model_mock.sim_telarray_seed = None
    array_model_mock.configuration_writers = {"sim_telarray": _sim_telarray_writer()}
    array_model_mock.telescope_models = {"LST-01": MagicMock()}
    array_model_mock.telescope_models["LST-01"].parameters = {}

    with patch("simtools.simtel.output_validation.read_sim_telarray_metadata") as read_meta_mock:
        read_meta_mock.return_value = ({}, {1: {}})

        with patch(
            "simtools.simtel.output_validation.get_sim_telarray_telescope_id"
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
    array_model_mock.configuration_writers = {"sim_telarray": _sim_telarray_writer()}
    array_model_mock.telescope_models = {"LST-01": MagicMock()}
    array_model_mock.telescope_models["LST-01"].parameters = {}

    with patch("simtools.simtel.output_validation.read_sim_telarray_metadata") as read_meta_mock:
        read_meta_mock.return_value = ({}, {1: {}})

        with patch(
            "simtools.simtel.output_validation.get_sim_telarray_telescope_id"
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

    with patch("simtools.simtel.output_validation.file_info.get_simulated_events") as mock_events:
        mock_events.return_value = mock_return
        if should_raise:
            with pytest.raises(ValueError, match="Event mismatch"):
                validate_sim_telarray_event_numbers([data_file], expected_mc, expected_shower)
        else:
            validate_sim_telarray_event_numbers([data_file], expected_mc, expected_shower)


def test_valid_log_files(tmp_path):
    log_file = tmp_path / "test.log"
    log_file.write_text("test log content")

    with patch("simtools.simtel.output_validation.check_plain_logs") as mock_check:
        mock_check.return_value = True
        assert validate_sim_telarray_logs([log_file]) is None


def test_invalid_log_files(tmp_path):
    log_file = tmp_path / "test.log"
    log_file.write_text("test log content")

    with patch("simtools.simtel.output_validation.check_plain_logs") as mock_check:
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

    with patch("simtools.simtel.output_validation.SimTelFile") as mock_file:
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

    with patch("simtools.simtel.output_validation.SimTelFile") as mock_file:
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

    with patch("simtools.simtel.output_validation.SimTelFile") as mock_file:
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

    with patch("simtools.simtel.output_validation.SimTelFile") as mock_file:
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

    with patch("simtools.simtel.output_validation.SimTelFile") as mock_file:
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
    with patch("simtools.simtel.output_validation.read_sim_telarray_metadata") as mock_read:
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

    with patch("simtools.simtel.output_validation.SimTelFile") as mock_file:
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

    with patch("simtools.simtel.output_validation.assert_sim_telarray_metadata") as mock_assert:
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

    with patch("simtools.simtel.output_validation.validate_sim_telarray_logs") as mock_log:
        with patch("simtools.simtel.output_validation.validate_sim_telarray_metadata") as mock_meta:
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
        patch("simtools.simtel.output_validation.validate_sim_telarray_logs"),
        patch(
            "simtools.simtel.output_validation.validate_sim_telarray_event_numbers"
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

    with patch("simtools.simtel.output_validation.SimTelFile") as mock_file:
        mock_file.return_value.__enter__.return_value = iter([{"type": "calibration"}])
        result = assert_events_of_type(data_file, event_type="shower")
        assert result is False
        assert "No events of type" in caplog.text


def _nsb_array_model(settings, written_nsb=None, exported=True):
    """Return an array model mock with one telescope and optionally recorded NSB settings."""
    array_model = MagicMock()
    array_model.model_version = "6.0.0"
    array_model.site_model.parameters = {"nsb_scaling_factor": {"value": 1.0, "type": "float64"}}
    array_model.site_model.get_simulation_software_parameters.return_value = {}
    array_model.sim_telarray_seed = None
    telescope = MagicMock()
    telescope.name = "LST-01"
    telescope.parameters = {
        "nsb_pixel_rate": {"value": 0.2, "type": "float64"},
        "nsb_autoscale_airmass": {"value": [0.84, 0.29], "type": "float64"},
    }
    telescope.configuration_writers = {
        "sim_telarray": _sim_telarray_writer(written_nsb=written_nsb)
    }
    array_model.telescope_models = {"LST-01": telescope}
    array_model.configuration_writers = {
        "sim_telarray": _sim_telarray_writer(settings=settings, exported=exported)
    }
    return array_model


def _written_nsb(pixel_rate):
    """Return NSB parameters as written to the telescope configuration."""
    return {
        "nsb_pixel_rate": {"value": pixel_rate},
        "nsb_autoscale_airmass": {"value": [0.0, 0.29]},
        "nsb_scaling_factor": {"value": 1.0},
    }


def _assert_nsb_metadata(tmp_path, array_model, pixel_rate, airmass):
    sim_file = tmp_path / "test.simtel.zst"
    sim_file.write_bytes(b"dummy")
    global_meta = {"nsb_scaling_factor": 1.0}
    telescope_meta = {"nsb_pixel_rate": pixel_rate, "nsb_autoscale_airmass": airmass}
    with (
        patch("simtools.simtel.output_validation.read_sim_telarray_metadata") as read_meta,
        patch("simtools.simtel.output_validation.get_sim_telarray_telescope_id", return_value=1),
        patch("simtools.simtel.output_validation.sim_telarray_metadata.validate_metadata_values"),
    ):
        read_meta.return_value = (global_meta, {1: telescope_meta})
        assert_sim_telarray_metadata(sim_file, array_model)


def test_assert_sim_telarray_metadata_accepts_recorded_nsb_scaling(tmp_path):
    settings = NsbSettings(zenith_angle=20.0, manual_factor=None, autoscale=True)
    expected_rate = 0.2 * airmass_scale(20.0, 0.84, 0.29)
    array_model = _nsb_array_model(settings, written_nsb=_written_nsb(expected_rate))

    _assert_nsb_metadata(tmp_path, array_model, pixel_rate=expected_rate, airmass="0.0 0.29")


def test_assert_sim_telarray_metadata_uses_recorded_nsb_values(tmp_path):
    settings = NsbSettings(zenith_angle=20.0, manual_factor=None, autoscale=True)
    array_model = _nsb_array_model(settings, written_nsb=_written_nsb(0.5))

    # recorded value is used even if it differs from a recomputation of the scaling
    _assert_nsb_metadata(tmp_path, array_model, pixel_rate=0.5, airmass="0.0 0.29")


def test_assert_sim_telarray_metadata_rejects_unscaled_nsb_rate(tmp_path):
    settings = NsbSettings(zenith_angle=20.0, manual_factor=None, autoscale=True)
    expected_rate = 0.2 * airmass_scale(20.0, 0.84, 0.29)
    array_model = _nsb_array_model(settings, written_nsb=_written_nsb(expected_rate))

    with pytest.raises(ValueError, match="nsb_pixel_rate"):
        _assert_nsb_metadata(tmp_path, array_model, pixel_rate=0.2, airmass="0.0 0.29")


def test_assert_sim_telarray_metadata_without_recorded_nsb_uses_model_values(tmp_path):
    array_model = _nsb_array_model(None)

    _assert_nsb_metadata(tmp_path, array_model, pixel_rate=0.2, airmass="0.84 0.29")


def test_assert_sim_telarray_metadata_with_settings_requires_recorded_telescope_nsb(tmp_path):
    settings = NsbSettings(zenith_angle=20.0, manual_factor=None, autoscale=True)
    array_model = _nsb_array_model(settings, written_nsb=None)

    with pytest.raises(ValueError, match="are unknown"):
        _assert_nsb_metadata(tmp_path, array_model, pixel_rate=0.2, airmass="0.0 0.29")


def test_assert_sim_telarray_metadata_requires_exported_configuration(tmp_path):
    array_model = _nsb_array_model(None, exported=False)

    with pytest.raises(ValueError, match="are unknown"):
        _assert_nsb_metadata(tmp_path, array_model, pixel_rate=0.2, airmass="0.84 0.29")


@pytest.mark.parametrize(
    "nsb_message",
    [
        "NSB level is auto-scaled",
        "NSB level is scaled by a factor",
        "Telescope has NSB scaling factor",
    ],
)
def test_validate_sim_telarray_logs_rejects_nsb_scaling_messages(tmp_test_directory, nsb_message):
    log_file = Path(tmp_test_directory) / "sim_telarray.log"
    log_file.write_text(f"Sim_telarray finished at now\nFinished.\n{nsb_message}\n")

    with pytest.raises(ValueError, match="validation failed"):
        validate_sim_telarray_logs([log_file])
