#!/usr/bin/python3

import copy
import re
from unittest.mock import MagicMock

import numpy as np
import pytest

from simtools.simtel.simtel_io_metadata import read_sim_telarray_metadata
from simtools.testing.sim_telarray_metadata import (
    _sim_telarray_name_from_parameter_name,
    assert_sim_telarray_metadata,
    is_equal,
)


@pytest.fixture
def test_metadata_file():
    return (
        "tests/resources/"
        "run000010_gamma_za20deg_azm000deg_North_test_layout_6.0.0"
        "_test-production-North.simtel.zst"
    )


@pytest.fixture
def test_metadata(test_metadata_file):
    return read_sim_telarray_metadata(test_metadata_file)


def test_assert_sim_telarray_metadata_success(test_metadata_file):
    """Test assert_sim_telarray_metadata with successful comparison."""
    array_model_mock = MagicMock()
    array_model_mock.site_model.parameters = {
        "altitude": {"value": 2158.0, "type": "float"},
    }
    array_model_mock.telescope_model = {}
    array_model_mock.telescope_model = {}
    array_model_mock.telescope_model["LST-01"] = MagicMock()
    array_model_mock.telescope_model["LST-01"].parameters = {
        "mirror_area": {"value": 386.0, "type": "float"},
    }
    array_model_mock.telescope_model["MST-01"] = MagicMock()
    array_model_mock.telescope_model["MST-01"].parameters = {
        "mirror_area": {"value": 386.0, "type": "float"},
    }
    array_model_mock.telescope_model["SST-01"] = MagicMock()
    array_model_mock.telescope_model["SST-01"].parameters = {
        "mirror_area": {"value": 386.0, "type": "float"},
    }

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Number of telescopes in sim_telarray file (13) does not match number of telescopes in array model (3)"
        ),
    ):
        assert_sim_telarray_metadata(test_metadata_file, array_model_mock)


def test_assert_sim_telarray_metadata_using_array_model(test_metadata_file, array_model_north):
    """Test assert_sim_telarray_metadata with different number of telescopes."""

    assert_sim_telarray_metadata(
        "tests/resources/"
        "run000010_gamma_za20deg_azm000deg_North_test_layout_6.0.0"
        "_test-production-North.simtel.zst",
        array_model_north,
    ) is None

    # rename one telescope
    array_model_north_renamed_telescope = copy.deepcopy(array_model_north)
    array_model_north_renamed_telescope.telescope_model = {
        f"tel_{old_key}": model
        for old_key, model in array_model_north_renamed_telescope.telescope_model.items()
    }

    with pytest.raises(
        ValueError, match=re.escape("Telescope tel_LSTN-01 not found in sim_telarray file metadata")
    ):
        assert_sim_telarray_metadata(test_metadata_file, array_model_north_renamed_telescope)


def test_assert_sim_telarray_metadata_with_mismatched_parameters(
    test_metadata_file, array_model_north
):
    """Test assert_sim_telarray_metadata with mismatched parameters."""
    array_model_mismatched_site = copy.deepcopy(array_model_north)
    array_model_mismatched_site.site_model.parameters["atmospheric_transmission"]["value"] = (
        "wrong_file.dat"
    )

    with pytest.raises(
        ValueError, match=r"^Site model parameters do not match sim_telarray metadata: "
    ):
        assert_sim_telarray_metadata(test_metadata_file, array_model_mismatched_site)

    array_model_mismatched_telescope = copy.deepcopy(array_model_north)
    array_model_mismatched_telescope.telescope_model["LSTN-02"].parameters[
        "random_mono_probability"
    ]["value"] = 0.99

    with pytest.raises(
        ValueError, match=r"^Telescope model parameters do not match sim_telarray metadata: "
    ):
        assert_sim_telarray_metadata(test_metadata_file, array_model_mismatched_telescope)


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
