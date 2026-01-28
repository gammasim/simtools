#!/usr/bin/python3

import logging
import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simtools.testing.sim_telarray_metadata import (
    _assert_model_parameters,
    _assert_sim_telarray_seed,
    _sim_telarray_name_from_parameter_name,
    assert_sim_telarray_metadata,
    is_equal,
)


def test_assert_sim_telarray_metadata(sim_telarray_file_gamma):
    """Test assert_sim_telarray_metadata."""
    array_model_mock = MagicMock()
    array_model_mock.site_model.parameters = {
        "altitude": {"value": 2158.0, "type": "float"},
    }
    array_model_mock.telescope_models = {}
    array_model_mock.telescope_models["LST-01"] = MagicMock()
    array_model_mock.telescope_models["LST-01"].parameters = {
        "mirror_area": {"value": 386.0, "type": "float"},
    }
    array_model_mock.telescope_models["MST-01"] = MagicMock()
    array_model_mock.telescope_models["MST-01"].parameters = {
        "mirror_area": {"value": 386.0, "type": "float"},
    }
    array_model_mock.telescope_models["SST-01"] = MagicMock()
    array_model_mock.telescope_models["SST-01"].parameters = {
        "mirror_area": {"value": 386.0, "type": "float"},
    }

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Number of telescopes in sim_telarray file (13)"
            " does not match number of telescopes in array model (3)"
        ),
    ):
        assert_sim_telarray_metadata(sim_telarray_file_gamma, array_model_mock)


def test_sim_telarray_name_from_parameter_name():
    """Test _sim_telarray_name_from_parameter_name."""
    assert _sim_telarray_name_from_parameter_name("reference_point_latitude") == "latitude"
    assert _sim_telarray_name_from_parameter_name("altitude") == "corsika_observation_level"
    assert _sim_telarray_name_from_parameter_name("array_triggers") is None


def test_is_equal():
    """Test is_equal."""
    assert is_equal(1.0, 1.0, "float") is True
    assert is_equal(1.0, 1.00000000001, "float") is True
    assert is_equal(1.0, 1.1, "float") is False
    assert is_equal("test", "test", "string") is True
    assert is_equal("test", "test ", "string") is True
    assert is_equal("test", "test1", "string") is False
    assert is_equal({"a": 1}, {"a": 1}, "dict") is True
    assert is_equal({"a": 1}, {"a": 2}, "dict") is False
    assert is_equal(True, True, "boolean") is True
    assert is_equal(True, False, "boolean") is False
    assert is_equal([1.0, 2.0], [1.0, 2.0], "list") is True
    assert is_equal([1.0, 2.0], [1.0, 2.1], "list") is False
    assert is_equal(1.0, [1.0, 1.0], "list") is True
    assert is_equal([1.0, 1.0], 1.0, "list") is True
    assert is_equal(np.array([1.0, 2.0]), np.array([1.0, 2.0]), "array") is True
    assert is_equal(np.array([1.0, 2.0]), np.array([1.0, 2.1]), "array") is False
    assert is_equal((1.0,), 1.0, "float") is True
    assert is_equal(None, None, "any") is True
    assert is_equal("none", None, "any") is True
    assert is_equal(None, "none", "any") is True


def test_assert_sim_telarray_seed(caplog):
    """Test _assert_sim_telarray_seed."""

    metadata = {"instrument_seed": "12345", "instrument_instances": 100}
    sim_telarray_seeds = MagicMock()
    sim_telarray_seeds.instrument_seed = "12345"

    # Test with matching seeds
    with caplog.at_level(logging.INFO):
        assert _assert_sim_telarray_seed(metadata, sim_telarray_seeds) is None
        assert "sim_telarray_seed in sim_telarray file: 12345, and model: 12345" in caplog.text

    # Test with mismatched seeds
    metadata = {"instrument_seed": "12345", "instrument_instances": 100}
    sim_telarray_seeds = MagicMock()
    sim_telarray_seeds.instrument_seed = "54321"
    assert (
        _assert_sim_telarray_seed(metadata, sim_telarray_seeds)
        == "Parameter instrument_seed mismatch between sim_telarray file: 12345, and model: 54321"
    )

    # Test with sim_telarray_seeds is None
    assert _assert_sim_telarray_seed(metadata, None) is None


def test_assert_sim_telarray_seed_with_rng_select_seed(caplog):
    """Test _assert_sim_telarray_seed with rng_select_seed."""
    metadata = {
        "instrument_seed": "12345",
        "instrument_instances": 100,
        "rng_select_seed": "12394",
    }
    sim_telarray_seeds = MagicMock()
    sim_telarray_seeds.instrument_seed = "12345"

    with patch(
        "simtools.testing.sim_telarray_metadata.get_corsika_run_number", return_value=10
    ) as get_corsika_run_number_mock:
        with caplog.at_level(logging.INFO):
            result = _assert_sim_telarray_seed(metadata, sim_telarray_seeds, "test_file")
            assert "Parameter rng_select_seed mismatch between sim_telarray file" in result
            assert "sim_telarray_seed in sim_telarray file: 12345, and model: 12345" in caplog.text
            get_corsika_run_number_mock.assert_called_once_with("test_file")


def test_assert_model_parameters():
    """Test _assert_model_parameters."""
    metadata = {
        "altitude": "2158.0",
        "mirror_area": "386.0",
        "camera_pixels": "1855",
    }

    model_mock = MagicMock()
    model_mock.parameters = {
        "altitude": {"value": 2158.0, "type": "float"},
        "mirror_area": {"value": 386.0, "type": "float"},
        "camera_pixels": {"value": 1855, "type": "int"},
    }

    with patch("simtools.testing.sim_telarray_metadata.SimtelConfigReader") as reader_mock:
        reader_instance = reader_mock.return_value
        reader_instance.extract_value_from_sim_telarray_column.side_effect = [
            (386.0, None),
            (1855, None),
            (2158.0, None),
        ]

        result = _assert_model_parameters(metadata, model_mock)
        assert len(result) == 0


def test_assert_model_parameters_with_mismatch():
    """Test _assert_model_parameters with mismatched values."""
    metadata = {
        "altitude": "2158.0",
        "mirror_area": "400.0",
    }

    model_mock = MagicMock()
    model_mock.parameters = {
        "altitude": {"value": 2158.0, "type": "float"},
        "mirror_area": {"value": 386.0, "type": "float"},
    }

    with patch("simtools.testing.sim_telarray_metadata.SimtelConfigReader") as reader_mock:
        reader_instance = reader_mock.return_value
        reader_instance.extract_value_from_sim_telarray_column.side_effect = [
            (400.0, None),
            (2158.0, None),
        ]

        result = _assert_model_parameters(metadata, model_mock)
        assert len(result) == 1
        assert "mirror_area" in result[0]
        assert "400.0" in result[0]
        assert "386.0" in result[0]


def test_assert_model_parameters_string_type():
    """Test _assert_model_parameters with string type."""
    metadata = {
        "telescope_name": "LST-01",
    }

    model_mock = MagicMock()
    model_mock.parameters = {
        "telescope_name": {"value": "LST-01", "type": "string"},
    }

    result = _assert_model_parameters(metadata, model_mock)
    assert len(result) == 0


def test_assert_model_parameters_missing_in_metadata():
    """Test _assert_model_parameters when parameter not in metadata."""
    metadata = {
        "mirror_area": "386.0",
    }

    model_mock = MagicMock()
    model_mock.parameters = {
        "altitude": {"value": 2158.0, "type": "float"},
        "mirror_area": {"value": 386.0, "type": "float"},
    }

    with patch("simtools.testing.sim_telarray_metadata.SimtelConfigReader") as reader_mock:
        reader_instance = reader_mock.return_value
        reader_instance.extract_value_from_sim_telarray_column.return_value = (386.0, None)

        result = _assert_model_parameters(metadata, model_mock)
        assert len(result) == 0


def test_assert_sim_telarray_metadata_telescope_not_found(sim_telarray_file_gamma):
    """Test assert_sim_telarray_metadata when telescope is not found in metadata."""
    array_model_mock = MagicMock()
    array_model_mock.site_model.parameters = {}
    array_model_mock.sim_telarray_seeds = None
    array_model_mock.telescope_models = {"LST-01": MagicMock()}
    array_model_mock.telescope_models["LST-01"].parameters = {}

    with patch(
        "simtools.testing.sim_telarray_metadata.read_sim_telarray_metadata"
    ) as read_meta_mock:
        read_meta_mock.return_value = ({}, {1: {}})

        with patch(
            "simtools.testing.sim_telarray_metadata.get_sim_telarray_telescope_id"
        ) as get_id_mock:
            get_id_mock.return_value = None

            with pytest.raises(
                ValueError, match="Telescope LST-01 not found in sim_telarray file metadata"
            ):
                assert_sim_telarray_metadata(sim_telarray_file_gamma, array_model_mock)


def test_assert_sim_telarray_metadata_parameter_mismatch(sim_telarray_file_gamma):
    """Test assert_sim_telarray_metadata with parameter mismatch."""
    array_model_mock = MagicMock()
    array_model_mock.site_model.parameters = {
        "altitude": {"value": 2000.0, "type": "float"},
    }
    array_model_mock.sim_telarray_seeds = None
    array_model_mock.telescope_models = {"TEL-01": MagicMock()}
    array_model_mock.telescope_models["TEL-01"].parameters = {}

    with patch(
        "simtools.testing.sim_telarray_metadata.read_sim_telarray_metadata"
    ) as read_meta_mock:
        read_meta_mock.return_value = ({"corsika_observation_level": "2158.0"}, {1: {}})

        with patch(
            "simtools.testing.sim_telarray_metadata.get_sim_telarray_telescope_id"
        ) as get_id_mock:
            get_id_mock.return_value = 1

            with patch("simtools.testing.sim_telarray_metadata.SimtelConfigReader") as reader_mock:
                reader_instance = reader_mock.return_value
                reader_instance.extract_value_from_sim_telarray_column.return_value = (2158.0, None)

                with pytest.raises(
                    ValueError, match="Telescope or site model parameters do not match"
                ):
                    assert_sim_telarray_metadata(sim_telarray_file_gamma, array_model_mock)
