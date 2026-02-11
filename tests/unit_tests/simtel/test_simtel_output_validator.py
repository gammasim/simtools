#!/usr/bin/python3

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simtools.simtel.simtel_output_validator import (
    _assert_model_parameters,
    _assert_sim_telarray_seed,
    _is_equal_floats_or_ints,
    _item_to_check_from_sim_telarray,
    _sim_telarray_name_from_parameter_name,
    assert_events_of_type,
    assert_expected_sim_telarray_metadata,
    assert_expected_sim_telarray_output,
    assert_n_showers_and_energy_range,
    assert_sim_telarray_metadata,
    is_equal,
    validate_event_numbers,
    validate_log_files,
    validate_metadata,
    validate_sim_telarray,
)


class TestSimTelArrayNameFromParameterName:
    """Test _sim_telarray_name_from_parameter_name function."""

    @pytest.mark.parametrize(
        ("input_name", "expected"),
        [
            ("reference_point_latitude", "latitude"),
            ("altitude", "corsika_observation_level"),
            ("array_triggers", None),
            ("mirror_area", "mirror_area"),
        ],
    )
    def test_parameter_name_conversion(self, input_name, expected):
        """Test parameter name conversion."""
        assert _sim_telarray_name_from_parameter_name(input_name) == expected


class TestIsEqual:
    """Test is_equal function with various types."""

    @pytest.mark.parametrize(
        ("value1", "value2", "value_type", "expected"),
        [
            (1.0, 1.0, "float", True),
            (1.0, 1.00000000001, "float", True),
            (1.0, 1.1, "float", False),
            ("test", "test", "string", True),
            ("test", "test ", "string", True),
            ("test", "test1", "string", False),
            ({"a": 1}, {"a": 1}, "dict", True),
            ({"a": 1}, {"a": 2}, "dict", False),
            (True, True, "boolean", True),
            (True, False, "boolean", False),
            ([1.0, 2.0], [1.0, 2.0], "list", True),
            ([1.0, 2.0], [1.0, 2.1], "list", False),
            (1.0, [1.0, 1.0], "list", True),
            ([1.0, 1.0], 1.0, "list", True),
            (np.array([1.0, 2.0]), np.array([1.0, 2.0]), "array", True),
            (np.array([1.0, 2.0]), np.array([1.0, 2.1]), "array", False),
            ((1.0,), 1.0, "float", True),
            (None, None, "any", True),
            ("none", None, "any", True),
            (None, "none", "any", True),
        ],
    )
    def test_is_equal(self, value1, value2, value_type, expected):
        """Test is_equal with various types and values."""
        assert is_equal(value1, value2, value_type) is expected


class TestIsEqualFloatsOrInts:
    """Test _is_equal_floats_or_ints function."""

    @pytest.mark.parametrize(
        ("value1", "value2", "expected"),
        [
            ([1.0, 2.0], [1.0, 2.0], True),
            (np.array([1.0, 2.0]), np.array([1.0, 2.0000000001]), True),
            ([1.0, 1.0], 1.0, True),
            (1.0, [1.0, 1.0], True),
            (1.0, 1.0000000001, True),
        ],
    )
    def test_is_equal_floats_or_ints(self, value1, value2, expected):
        """Test _is_equal_floats_or_ints function."""
        result = _is_equal_floats_or_ints(value1, value2)
        assert result is expected


class TestAssertSimTelArraySeed:
    """Test _assert_sim_telarray_seed function."""

    @pytest.mark.parametrize(
        ("metadata", "seed_value", "expected_result"),
        [
            ({"instrument_seed": "12345"}, None, None),
            ({}, MagicMock(instrument_seed="12345"), None),
        ],
    )
    def test_seed_returns_none(self, metadata, seed_value, expected_result):
        """Test cases where result should be None."""
        result = _assert_sim_telarray_seed(metadata, seed_value)
        assert result is expected_result

    def test_matching_seeds(self, caplog):
        """Test matching seeds."""
        metadata = {"instrument_seed": "12345", "instrument_instances": 100}
        sim_telarray_seeds = MagicMock()
        sim_telarray_seeds.instrument_seed = "12345"

        with caplog.at_level(logging.INFO):
            result = _assert_sim_telarray_seed(metadata, sim_telarray_seeds)
            assert result is None
            assert "sim_telarray_seed" in caplog.text

    def test_mismatched_seeds(self):
        """Test mismatched seeds."""
        metadata = {"instrument_seed": "12345", "instrument_instances": 100}
        sim_telarray_seeds = MagicMock()
        sim_telarray_seeds.instrument_seed = "54321"

        result = _assert_sim_telarray_seed(metadata, sim_telarray_seeds)
        assert result is not None
        assert "instrument_seed" in result
        assert "12345" in result
        assert "54321" in result

    def test_rng_select_seed_mismatch(self):
        """Test rng_select_seed mismatch."""
        metadata = {
            "instrument_seed": "12345",
            "instrument_instances": "100",
            "rng_select_seed": "99999",
        }
        sim_telarray_seeds = MagicMock()
        sim_telarray_seeds.instrument_seed = "12345"

        with patch(
            "simtools.simtel.simtel_output_validator.get_corsika_run_number", return_value=10
        ) as mock_get_run:
            test_seeds = list(range(100))
            with patch(
                "simtools.simtel.simtel_output_validator.random.seeds", return_value=test_seeds
            ):
                result = _assert_sim_telarray_seed(
                    metadata, sim_telarray_seeds, Path("/nonexistent/test_file")
                )
                assert result is not None
                assert "rng_select_seed" in result
                mock_get_run.assert_called_once()


class TestAssertModelParameters:
    """Test _assert_model_parameters function."""

    @pytest.mark.parametrize(
        (
            "metadata",
            "parameters",
            "mock_value",
            "expected_len",
            "expected_in_result",
            "allow_changes",
        ),
        [
            (
                {"mirror_area": "386.0"},
                {"mirror_area": {"value": 386.0, "type": "float"}},
                (386.0, None),
                0,
                None,
                None,
            ),
            (
                {"mirror_area": "400.0"},
                {"mirror_area": {"value": 386.0, "type": "float"}},
                (400.0, None),
                1,
                "mirror_area",
                None,
            ),
            (
                {"mirror_area": "400.0"},
                {"mirror_area": {"value": 386.0, "type": "float"}},
                (400.0, None),
                0,
                None,
                ["mirror_area"],
            ),
        ],
    )
    def test_assert_model_parameters(
        self, metadata, parameters, mock_value, expected_len, expected_in_result, allow_changes
    ):
        """Test model parameters matching."""
        model_mock = MagicMock()
        model_mock.parameters = parameters

        with patch("simtools.simtel.simtel_config_reader.SimtelConfigReader") as reader_mock:
            reader_instance = reader_mock.return_value
            reader_instance.extract_value_from_sim_telarray_column.return_value = mock_value

            result = _assert_model_parameters(metadata, model_mock, allow_for_changes=allow_changes)
            assert len(result) == expected_len
            if expected_in_result:
                assert expected_in_result in result[0]

    def test_string_parameter(self):
        """Test string type parameter."""
        metadata = {"telescope_name": "LST-01"}
        model_mock = MagicMock()
        model_mock.parameters = {
            "telescope_name": {"value": "LST-01", "type": "string"},
        }

        result = _assert_model_parameters(metadata, model_mock)
        assert len(result) == 0

    def test_missing_parameter_in_metadata(self):
        """Test parameter not in metadata."""
        metadata = {}
        model_mock = MagicMock()
        model_mock.parameters = {
            "mirror_area": {"value": 386.0, "type": "float"},
        }

        result = _assert_model_parameters(metadata, model_mock)
        assert len(result) == 0


class TestAssertSimTelArrayMetadata:
    """Test assert_sim_telarray_metadata function."""

    def test_telescope_count_mismatch(self, tmp_path):
        """Test error when telescope count mismatches."""
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
            "simtools.simtel.simtel_output_validator.read_sim_telarray_metadata"
        ) as read_meta_mock:
            read_meta_mock.return_value = ({}, {1: {}, 2: {}, 3: {}})

            with pytest.raises(ValueError, match="Number of telescopes"):
                assert_sim_telarray_metadata(sim_file, array_model_mock)

    def test_telescope_not_found(self, tmp_path):
        """Test error when telescope not found."""
        sim_file = tmp_path / "test.simtel.zst"
        sim_file.write_bytes(b"dummy")

        array_model_mock = MagicMock()
        array_model_mock.site_model.parameters = {}
        array_model_mock.sim_telarray_seed = None
        array_model_mock.telescope_models = {"LST-01": MagicMock()}
        array_model_mock.telescope_models["LST-01"].parameters = {}

        with patch(
            "simtools.simtel.simtel_output_validator.read_sim_telarray_metadata"
        ) as read_meta_mock:
            read_meta_mock.return_value = ({}, {1: {}})

            with patch(
                "simtools.simtel.simtel_output_validator.get_sim_telarray_telescope_id"
            ) as get_id_mock:
                get_id_mock.return_value = None

                with pytest.raises(ValueError, match="not found"):
                    assert_sim_telarray_metadata(sim_file, array_model_mock)

    def test_valid_metadata(self, tmp_path):
        """Test valid metadata passes."""
        sim_file = tmp_path / "test.simtel.zst"
        sim_file.write_bytes(b"dummy")

        array_model_mock = MagicMock()
        array_model_mock.site_model.parameters = {}
        array_model_mock.sim_telarray_seed = None
        array_model_mock.telescope_models = {"LST-01": MagicMock()}
        array_model_mock.telescope_models["LST-01"].parameters = {}

        with patch(
            "simtools.simtel.simtel_output_validator.read_sim_telarray_metadata"
        ) as read_meta_mock:
            read_meta_mock.return_value = ({}, {1: {}})

            with patch(
                "simtools.simtel.simtel_output_validator.get_sim_telarray_telescope_id"
            ) as get_id_mock:
                get_id_mock.return_value = 1

                assert_sim_telarray_metadata(sim_file, array_model_mock)


class TestValidateEventNumbers:
    """Test validate_event_numbers function."""

    @pytest.mark.parametrize(
        ("mock_return", "expected_mc", "expected_shower", "should_raise"),
        [
            ((100, 1000), 1000, 100, False),
            ((50, 500), 1000, 100, True),
        ],
    )
    def test_validate_event_numbers(
        self, tmp_path, mock_return, expected_mc, expected_shower, should_raise
    ):
        """Test event number validation."""
        data_file = tmp_path / "test.simtel.zst"
        data_file.write_bytes(b"dummy")

        with patch("simtools.sim_events.file_info.get_simulated_events") as mock_events:
            mock_events.return_value = mock_return
            if should_raise:
                with pytest.raises(ValueError, match="Event mismatch"):
                    validate_event_numbers([data_file], expected_mc, expected_shower)
            else:
                validate_event_numbers([data_file], expected_mc, expected_shower)


class TestValidateLogFiles:
    """Test validate_log_files function."""

    def test_valid_log_files(self, tmp_path):
        """Test valid log files pass."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test log content")

        with patch("simtools.simtel.simtel_output_validator.check_plain_logs") as mock_check:
            mock_check.return_value = True
            validate_log_files([log_file])

    def test_invalid_log_files(self, tmp_path):
        """Test invalid log files raise error."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test log content")

        with patch("simtools.simtel.simtel_output_validator.check_plain_logs") as mock_check:
            mock_check.return_value = False
            with pytest.raises(ValueError, match="validation failed"):
                validate_log_files([log_file])


class TestAssertNShowersAndEnergyRange:
    """Test assert_n_showers_and_energy_range function."""

    def test_valid_showers_and_energy(self, tmp_path):
        """Test valid shower count and energy range."""
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

        with patch("simtools.simtel.simtel_output_validator.SimTelFile") as mock_file:
            mock_instance = MagicMock()
            mock_instance.__enter__.return_value = mock_instance
            mock_instance.__exit__.return_value = None
            mock_instance.mc_run_headers = [mock_run_header]
            mock_instance.__iter__.return_value = [mock_event] * 99
            mock_file.return_value = mock_instance

            result = assert_n_showers_and_energy_range(sim_file)
            assert result is True

    def test_shower_count_mismatch(self, tmp_path):
        """Test shower count mismatch raises error."""
        sim_file = tmp_path / "test.simtel.zst"
        sim_file.write_bytes(b"dummy")

        mock_event = {"mc_shower": {"energy": 1.0}, "type": "data"}
        mock_run_header = {"n_showers": 1000, "E_range": [0.5, 10.0]}

        with patch("simtools.simtel.simtel_output_validator.SimTelFile") as mock_file:
            mock_instance = MagicMock()
            mock_instance.__enter__.return_value = mock_instance
            mock_instance.__exit__.return_value = None
            mock_instance.mc_run_headers = [mock_run_header]
            mock_instance.__iter__.return_value = [mock_event] * 50
            mock_file.return_value = mock_instance

            with pytest.raises(ValueError, match="does not match"):
                assert_n_showers_and_energy_range(sim_file)

    def test_energy_range_exceeded(self, tmp_path):
        """Test energy outside configured range raises error."""
        sim_file = tmp_path / "test.simtel.zst"
        sim_file.write_bytes(b"dummy")

        mock_event = {"mc_shower": {"energy": 20.0}, "type": "data"}
        mock_run_header = {"n_showers": 100, "E_range": [0.5, 10.0]}

        with patch("simtools.simtel.simtel_output_validator.SimTelFile") as mock_file:
            mock_instance = MagicMock()
            mock_instance.__enter__.return_value = mock_instance
            mock_instance.__exit__.return_value = None
            mock_instance.mc_run_headers = [mock_run_header]
            mock_instance.__iter__.return_value = [mock_event] * 100
            mock_file.return_value = mock_instance

            with pytest.raises(ValueError, match="Energy range"):
                assert_n_showers_and_energy_range(sim_file)


class TestItemToCheckFromSimTelArray:
    """Test _item_to_check_from_sim_telarray function."""

    def test_extract_telescope_events(self, tmp_path):
        """Test extraction of telescope events."""
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

        with patch("simtools.simtel.simtel_output_validator.SimTelFile") as mock_file:
            mock_instance = MagicMock()
            mock_instance.__enter__.return_value = mock_instance
            mock_instance.__exit__.return_value = None
            mock_instance.__iter__.return_value = [mock_event]
            mock_file.return_value = mock_instance

            result = _item_to_check_from_sim_telarray(sim_file, expected_output)

            assert "n_telescope_events" in result
            assert result["n_telescope_events"] == 1


class TestAssertExpectedSimTelArrayOutput:
    """Test assert_expected_sim_telarray_output function."""

    def test_none_expected_output(self):
        """Test None expected output returns True."""
        result = assert_expected_sim_telarray_output(Path("/dummy/file.zst"), None)
        assert result is True

    @pytest.mark.parametrize(
        ("expected_output", "extracted_data", "expected_result"),
        [
            ({}, {"n_telescope_events": 0, "n_calibration_events": 0}, True),
            (
                {"pe_sum": [10.0, 100.0]},
                {"pe_sum": [50.0], "n_telescope_events": 0, "n_calibration_events": 0},
                True,
            ),
            (
                {"pe_sum": [100.0, 200.0]},
                {"pe_sum": [50.0], "n_telescope_events": 0, "n_calibration_events": 0},
                False,
            ),
        ],
    )
    def test_expected_sim_telarray_output(self, expected_output, extracted_data, expected_result):
        """Test sim_telarray output validation."""
        with patch(
            "simtools.simtel.simtel_output_validator._item_to_check_from_sim_telarray"
        ) as mock_extract:
            mock_extract.return_value = extracted_data
            result = assert_expected_sim_telarray_output(Path("/dummy/file.zst"), expected_output)
            assert result is expected_result


class TestAssertExpectedSimTelArrayMetadata:
    """Test assert_expected_sim_telarray_metadata function."""

    def test_none_expected_metadata(self):
        """Test None expected metadata returns True."""
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
    def test_assert_expected_sim_telarray_metadata(
        self, expected_metadata, metadata_return, expected_result
    ):
        """Test sim_telarray metadata validation."""
        with patch(
            "simtools.simtel.simtel_output_validator.read_sim_telarray_metadata"
        ) as mock_read:
            mock_read.return_value = metadata_return
            result = assert_expected_sim_telarray_metadata(
                Path("/dummy/file.zst"), expected_metadata
            )
            assert result is expected_result


class TestAssertEventsOfType:
    """Test assert_events_of_type function."""

    @pytest.mark.parametrize(
        ("mock_events", "event_type", "expected_result"),
        [
            ([{"type": "data"}], "shower", True),
            ([{"type": "calibration"}], "pedestal", True),
            ([], "shower", False),
        ],
    )
    def test_assert_events_of_type(self, tmp_path, mock_events, event_type, expected_result):
        """Test event type assertion."""
        sim_file = tmp_path / "test.simtel.zst"
        sim_file.write_bytes(b"dummy")

        with patch("simtools.simtel.simtel_output_validator.SimTelFile") as mock_file:
            mock_instance = MagicMock()
            mock_instance.__enter__.return_value = mock_instance
            mock_instance.__exit__.return_value = None
            mock_instance.__iter__.return_value = mock_events
            mock_file.return_value = mock_instance

            result = assert_events_of_type(sim_file, event_type)
            assert result is expected_result


class TestValidateMetadata:
    """Test validate_metadata function."""

    @pytest.mark.parametrize(
        ("filename", "model_version", "should_call_assert"),
        [
            ("model_v1.0.0.simtel.zst", "1.0.0", True),
            ("other_v2.0.0.simtel.zst", "1.0.0", False),
        ],
    )
    def test_validate_metadata(self, tmp_path, filename, model_version, should_call_assert):
        """Test metadata validation."""
        test_file = tmp_path / filename
        test_file.write_bytes(b"dummy")

        model_mock = MagicMock()
        model_mock.model_version = model_version

        with patch(
            "simtools.simtel.simtel_output_validator.assert_sim_telarray_metadata"
        ) as mock_assert:
            validate_metadata([test_file], [model_mock])
            if should_call_assert:
                mock_assert.assert_called_once()
            else:
                mock_assert.assert_not_called()


class TestValidateSimTelArray:
    """Test validate_sim_telarray integration function."""

    @pytest.mark.parametrize(
        ("array_models", "should_call_metadata"),
        [
            (None, False),
            ([MagicMock()], True),
        ],
    )
    def test_validate_sim_telarray(self, tmp_path, array_models, should_call_metadata):
        """Test sim_telarray validation."""
        data_file = tmp_path / "test.simtel.zst"
        data_file.write_bytes(b"dummy")
        log_file = tmp_path / "test.log"
        log_file.write_text("test log")

        with patch("simtools.simtel.simtel_output_validator.validate_log_files") as mock_log:
            with patch("simtools.simtel.simtel_output_validator.validate_metadata") as mock_meta:
                validate_sim_telarray([data_file], [log_file], array_models=array_models)
                mock_log.assert_called_once()
                if should_call_metadata:
                    mock_meta.assert_called_once()
                else:
                    mock_meta.assert_not_called()


class TestValidateMetadataCoverage:
    """Test cases to improve coverage of validate_metadata."""

    def test_validate_metadata_with_matching_file(self, tmp_path, caplog):
        """Test validate_metadata logging when file matches."""
        caplog.set_level(logging.INFO)
        data_file = tmp_path / "model_v1.0.0.simtel.zst"
        data_file.write_bytes(b"dummy")

        model = MagicMock()
        model.model_version = "1.0.0"
        model.site_model = MagicMock()
        model.site_model.parameters = {}
        model.telescope_models = {}
        model.sim_telarray_seed = None

        with patch(
            "simtools.simtel.simtel_output_validator.read_sim_telarray_metadata"
        ) as mock_read:
            mock_read.return_value = ({}, {})
            validate_metadata([data_file], [model])
            assert "Validating metadata for" in caplog.text

    def test_validate_metadata_no_matching_file(self, tmp_path, caplog):
        """Test validate_metadata logging when no file matches."""
        caplog.set_level(logging.WARNING)
        data_file = tmp_path / "other_v1.0.0.simtel.zst"
        data_file.write_bytes(b"dummy")

        model = MagicMock()
        model.model_version = "2.0.0"

        validate_metadata([data_file], [model])
        assert "No sim_telarray file found" in caplog.text


class TestAssertSimTelArraySeedCoverage:
    """Test cases for seed assertions."""

    def test_assert_sim_telarray_seed_matching_seed(self, tmp_path, caplog):
        """Test seed assertion when seeds match."""
        caplog.set_level(logging.INFO)
        data_file = tmp_path / "test.simtel.zst"
        data_file.write_bytes(b"dummy")

        metadata = {
            "instrument_seed": "12345",
            "instrument_instances": "10",
            "rng_select_seed": "0",
        }
        seed = MagicMock()
        seed.instrument_seed = "12345"

        with patch("simtools.simtel.simtel_output_validator.get_corsika_run_number") as mock_run:
            with patch("simtools.simtel.simtel_output_validator.random.seeds") as mock_seeds:
                mock_run.return_value = 1
                mock_seeds.return_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                result = _assert_sim_telarray_seed(metadata, seed, data_file)
                assert result is None
                assert "sim_telarray_seed in sim_telarray file" in caplog.text


class TestAssertExpectedOutputCoverage:
    """Test cases for expected output assertions."""

    def test_assert_expected_output_no_data_found(self, tmp_path, caplog):
        """Test when no data is found for expected key."""
        caplog.set_level(logging.ERROR)
        data_file = tmp_path / "test.simtel.zst"
        data_file.write_bytes(b"dummy")

        expected_output = {"pe_sum": (0, 100)}

        with patch(
            "simtools.simtel.simtel_output_validator._item_to_check_from_sim_telarray"
        ) as mock_item:
            mock_item.return_value = {
                "n_telescope_events": 0,
                "n_calibration_events": 0,
                "pe_sum": [],
            }
            result = assert_expected_sim_telarray_output(data_file, expected_output)
            assert result is False
            assert "No data found" in caplog.text

    def test_assert_expected_output_mean_out_of_range(self, tmp_path, caplog):
        """Test when mean is out of expected range."""
        caplog.set_level(logging.ERROR)
        data_file = tmp_path / "test.simtel.zst"
        data_file.write_bytes(b"dummy")

        expected_output = {"pe_sum": (100, 200)}

        with patch(
            "simtools.simtel.simtel_output_validator._item_to_check_from_sim_telarray"
        ) as mock_item:
            mock_item.return_value = {
                "n_telescope_events": 1,
                "n_calibration_events": 0,
                "pe_sum": [10.0, 20.0],
            }
            result = assert_expected_sim_telarray_output(data_file, expected_output)
            assert result is False
            assert "not in the expected range" in caplog.text


class TestAssertExpectedMetadataCoverage:
    """Test cases for expected metadata assertions."""

    def test_assert_expected_metadata_matching_key(self, tmp_path, caplog):
        """Test when metadata key matches."""
        caplog.set_level(logging.DEBUG)
        data_file = tmp_path / "test.simtel.zst"
        data_file.write_bytes(b"dummy")

        expected_metadata = {"test_key": "test_value"}

        with patch(
            "simtools.simtel.simtel_output_validator.read_sim_telarray_metadata"
        ) as mock_read:
            mock_read.return_value = ({"test_key": "test_value"}, {})
            result = assert_expected_sim_telarray_metadata(data_file, expected_metadata)
            assert result is True
            assert "matches expected value" in caplog.text


class TestAssertEventsOfTypeCoverage:
    """Test cases for event type assertions."""

    def test_assert_events_of_type_not_found(self, tmp_path, caplog):
        """Test when no events of expected type are found."""
        caplog.set_level(logging.ERROR)
        data_file = tmp_path / "test.simtel.zst"
        data_file.write_bytes(b"dummy")

        with patch("simtools.simtel.simtel_output_validator.SimTelFile") as mock_file:
            mock_file.return_value.__enter__.return_value = iter([{"type": "calibration"}])
            result = assert_events_of_type(data_file, event_type="shower")
            assert result is False
            assert "No events of type" in caplog.text


class TestAssertNShowersEnergyRange:
    """Test assert_n_showers_and_energy_range."""

    def test_assert_n_showers_pass(self, tmp_path):
        """Test successful shower count validation."""
        data_file = tmp_path / "test.simtel.zst"
        data_file.write_bytes(b"dummy")

        with patch("simtools.simtel.simtel_output_validator.SimTelFile") as mock_file:
            mock_instance = MagicMock()
            mock_instance.__enter__.return_value = mock_instance
            mock_instance.__exit__.return_value = None
            mock_instance.mc_run_headers = [{"n_showers": 100, "E_range": [0.01, 100.0]}]
            mock_instance.__iter__.return_value = [
                {"mc_shower": {"energy": 1.0}},
                {"mc_shower": {"energy": 10.0}},
                {"mc_shower": {"energy": 50.0}},
            ] * 33 + [{"mc_shower": {"energy": 5.0}}]
            mock_file.return_value = mock_instance

            result = assert_n_showers_and_energy_range(data_file)
            assert result is True


class TestAssertSimTelArrayMetadataEdgeCases:
    """Test edge cases for metadata assertions."""

    def test_assert_sim_telarray_metadata_no_seed(self, tmp_path):
        """Test metadata assertion when sim_telarray_seed is None."""
        data_file = tmp_path / "test.simtel.zst"
        data_file.write_bytes(b"dummy")

        model = MagicMock()
        model.site_model = MagicMock()
        model.site_model.parameters = {}
        model.telescope_models = {}
        model.sim_telarray_seed = None

        with patch(
            "simtools.simtel.simtel_output_validator.read_sim_telarray_metadata"
        ) as mock_read:
            mock_read.return_value = ({}, {})
            # Should not raise an error
            assert_sim_telarray_metadata(data_file, model)
