#!/usr/bin/python3

import copy
import logging
import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simtools.testing.sim_telarray_metadata import (
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


def test_assert_sim_telarray_metadata_using_array_model(sim_telarray_file_gamma):
    """Test assert_sim_telarray_metadata with different number of telescopes."""
    # Create a mock array model matching the actual file values
    array_model_mock = MagicMock()
    array_model_mock.site_model.parameters = {
        "altitude": {"value": 2147.0, "type": "float"},
        "atmospheric_transmission": {
            "value": "atm_trans_2156_1_3_2_0_0_0.1_0.1.dat",
            "type": "string",
        },
    }

    # Create telescopes matching the test file - 4 LSTN and 9 MSTN
    array_model_mock.telescope_models = {}
    lstn_names = ["LSTN-01", "LSTN-02", "LSTN-03", "LSTN-04"]
    mstn_names = [
        "MSTN-01",
        "MSTN-02",
        "MSTN-03",
        "MSTN-04",
        "MSTN-05",
        "MSTN-06",
        "MSTN-07",
        "MSTN-10",
        "MSTN-15",
    ]
    for tel_name in lstn_names:
        array_model_mock.telescope_models[tel_name] = MagicMock(
            parameters={
                "random_mono_probability": {"value": 0.01, "type": "float"},
                "mirror_area": {"value": 386.97, "type": "float"},
            }
        )
    for tel_name in mstn_names:
        array_model_mock.telescope_models[tel_name] = MagicMock(
            parameters={
                "random_mono_probability": {"value": 0.01, "type": "float"},
                "mirror_area": {"value": 106.26, "type": "float"},
            }
        )

    # Test with matching parameters
    assert_sim_telarray_metadata(sim_telarray_file_gamma, array_model_mock) is None

    # Test with renamed telescope (adding different_ prefix)
    array_model_renamed = copy.deepcopy(array_model_mock)
    array_model_renamed.telescope_models = {
        f"different_{key}": model for key, model in array_model_mock.telescope_models.items()
    }
    error_msg = "Telescope different_LSTN-01 not found in sim_telarray file metadata"
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        assert_sim_telarray_metadata(sim_telarray_file_gamma, array_model_renamed)


def test_assert_sim_telarray_metadata_with_mismatched_parameters(sim_telarray_file_gamma):
    """Test assert_sim_telarray_metadata with mismatched parameters."""
    # Create base mock array model matching file
    array_model_mock = MagicMock()
    array_model_mock.site_model.parameters = {
        "altitude": {"value": 2147.0, "type": "float"},
        "atmospheric_transmission": {
            "value": "atm_trans_2156_1_3_2_0_0_0.1_0.1.dat",
            "type": "string",
        },
    }

    # Create telescopes matching the test file - 4 LSTN and 9 MSTN
    array_model_mock.telescope_models = {}
    lstn_names = ["LSTN-01", "LSTN-02", "LSTN-03", "LSTN-04"]
    mstn_names = [
        "MSTN-01",
        "MSTN-02",
        "MSTN-03",
        "MSTN-04",
        "MSTN-05",
        "MSTN-06",
        "MSTN-07",
        "MSTN-10",
        "MSTN-15",
    ]
    for tel_name in lstn_names:
        array_model_mock.telescope_models[tel_name] = MagicMock(
            parameters={
                "random_mono_probability": {"value": 0.01, "type": "float"},
                "mirror_area": {"value": 386.97, "type": "float"},
            }
        )
    for tel_name in mstn_names:
        array_model_mock.telescope_models[tel_name] = MagicMock(
            parameters={
                "random_mono_probability": {"value": 0.01, "type": "float"},
                "mirror_area": {"value": 106.26, "type": "float"},
            }
        )

    # Test with mismatched site parameter
    array_model_mismatched = copy.deepcopy(array_model_mock)
    array_model_mismatched.site_model.parameters["atmospheric_transmission"]["value"] = (
        "wrong_file.dat"
    )

    mismatch_message = r"^Telescope or site model parameters do not match sim_telarray"
    with pytest.raises(ValueError, match=mismatch_message):
        assert_sim_telarray_metadata(sim_telarray_file_gamma, array_model_mismatched)

    # Test with mismatched telescope parameter
    array_model_mismatched_telescope = copy.deepcopy(array_model_mock)
    array_model_mismatched_telescope.telescope_models["MSTN-01"].parameters[
        "random_mono_probability"
    ]["value"] = 0.99

    with pytest.raises(ValueError, match=mismatch_message):
        assert_sim_telarray_metadata(sim_telarray_file_gamma, array_model_mismatched_telescope)


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
    sim_telarray_seeds = {"seed": "12345", "instrument_instances": 100}

    # Test with matching seeds
    with caplog.at_level(logging.INFO):
        assert _assert_sim_telarray_seed(metadata, sim_telarray_seeds) is None
        assert "sim_telarray_seed in sim_telarray file: 12345, and model: 12345" in caplog.text

    # Test with mismatched seeds
    metadata = {"instrument_seed": "12345", "instrument_instances": 100}
    sim_telarray_seeds = {"seed": "54321", "instrument_instances": 100}
    assert (
        _assert_sim_telarray_seed(metadata, sim_telarray_seeds)
        == "Parameter instrument_seed mismatch between sim_telarray file: 12345, and model: 54321"
    )

    # Test with sim_telarray_seeds is None
    assert _assert_sim_telarray_seed(metadata, None) is None


def test_assert_sim_telarray_metadata_seed_mismatch(sim_telarray_file_gamma):
    """Test assert_sim_telarray_metadata with mismatched seeds."""
    # Create a mock array model
    array_model_mock = MagicMock()
    array_model_mock.site_model.parameters = {
        "altitude": {"value": 2147.0, "type": "float"},
        "atmospheric_transmission": {
            "value": "atm_trans_2156_1_3_2_0_0_0.1_0.1.dat",
            "type": "string",
        },
    }

    # Create telescopes matching the test file - 4 LSTN and 9 MSTN
    array_model_mock.telescope_models = {}
    lstn_names = ["LSTN-01", "LSTN-02", "LSTN-03", "LSTN-04"]
    mstn_names = [
        "MSTN-01",
        "MSTN-02",
        "MSTN-03",
        "MSTN-04",
        "MSTN-05",
        "MSTN-06",
        "MSTN-07",
        "MSTN-10",
        "MSTN-15",
    ]
    for tel_name in lstn_names:
        array_model_mock.telescope_models[tel_name] = MagicMock(
            parameters={
                "random_mono_probability": {"value": 0.01, "type": "float"},
                "mirror_area": {"value": 386.97, "type": "float"},
            }
        )
    for tel_name in mstn_names:
        array_model_mock.telescope_models[tel_name] = MagicMock(
            parameters={
                "random_mono_probability": {"value": 0.01, "type": "float"},
                "mirror_area": {"value": 106.26, "type": "float"},
            }
        )

    array_model_mock.sim_telarray_seeds = {"seed": "54321"}

    with patch(
        "simtools.testing.sim_telarray_metadata._assert_sim_telarray_seed",
        return_value="Parameter sim_telarray_seeds mismatch",
    ):
        mismatch_msg = "^Telescope or site model parameters do not match sim_telarray"
        with pytest.raises(ValueError, match=mismatch_msg):
            assert_sim_telarray_metadata(sim_telarray_file_gamma, array_model_mock)


def test_assert_sim_telarray_seed_with_rng_select_seed(caplog):
    """Test _assert_sim_telarray_seed with rng_select_seed."""
    metadata = {
        "instrument_seed": "12345",
        "instrument_instances": 100,
        "rng_select_seed": "12394",
    }
    sim_telarray_seeds = {"seed": "12345", "instrument_instances": 100}

    with patch(
        "simtools.testing.sim_telarray_metadata.get_corsika_run_number", return_value=10
    ) as get_corsika_run_number_mock:
        with caplog.at_level(logging.INFO):
            result = _assert_sim_telarray_seed(metadata, sim_telarray_seeds, "test_file")
            assert "Parameter rng_select_seed mismatch between sim_telarray file" in result
            assert "sim_telarray_seed in sim_telarray file: 12345, and model: 12345" in caplog.text
            get_corsika_run_number_mock.assert_called_once_with("test_file")
