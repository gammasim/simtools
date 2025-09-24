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


def test_assert_sim_telarray_metadata_using_array_model(sim_telarray_file_gamma, array_model_north):
    """Test assert_sim_telarray_metadata with different number of telescopes."""

    assert_sim_telarray_metadata(sim_telarray_file_gamma, array_model_north) is None

    # rename one telescope
    array_model_north_renamed_telescope = copy.deepcopy(array_model_north)
    array_model_north_renamed_telescope.telescope_models = {
        f"tel_{old_key}": model
        for old_key, model in array_model_north_renamed_telescope.telescope_models.items()
    }

    with pytest.raises(
        ValueError, match=re.escape("Telescope tel_LSTN-01 not found in sim_telarray file metadata")
    ):
        assert_sim_telarray_metadata(sim_telarray_file_gamma, array_model_north_renamed_telescope)


def test_assert_sim_telarray_metadata_with_mismatched_parameters(
    sim_telarray_file_gamma, array_model_north
):
    """Test assert_sim_telarray_metadata with mismatched parameters."""
    array_model_mismatched_site = copy.deepcopy(array_model_north)
    array_model_mismatched_site.site_model.parameters["atmospheric_transmission"]["value"] = (
        "wrong_file.dat"
    )

    mismatch_message = r"^Telescope or site model parameters do not match sim_telarray "
    with pytest.raises(ValueError, match=mismatch_message):
        assert_sim_telarray_metadata(sim_telarray_file_gamma, array_model_mismatched_site)

    array_model_mismatched_telescope = copy.deepcopy(array_model_north)
    array_model_mismatched_telescope.telescope_models["LSTN-02"].parameters[
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


def test_assert_sim_telarray_metadata_seed_mismatch(sim_telarray_file_gamma, array_model_north):
    """Test assert_sim_telarray_metadata with mismatched seeds."""
    array_model_mismatched_seed = copy.deepcopy(array_model_north)
    array_model_mismatched_seed.sim_telarray_seeds = {"seed": "54321"}

    with patch(
        "simtools.testing.sim_telarray_metadata._assert_sim_telarray_seed",
        return_value="Parameter sim_telarray_seeds mismatch",
    ):
        with pytest.raises(
            ValueError, match=r"^Telescope or site model parameters do not match sim_telarray "
        ):
            assert_sim_telarray_metadata(sim_telarray_file_gamma, array_model_mismatched_seed)


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
