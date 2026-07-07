#!/usr/bin/python3

import pytest

from simtools.simtel import simtel_validate_metadata


def test_get_meta_parameter_registry_uses_generated_and_model_parameter_sources():
    registry = simtel_validate_metadata.get_meta_parameter_registry()["meta_parameters"]

    assert registry["simtools_version"]["source_type"] == "generated"
    assert registry["random_mono_prob"]["source_type"] == "model_parameter"


def test_get_meta_parameter_registry_uses_add_as_presence_only():
    registry = simtel_validate_metadata.get_meta_parameter_registry()["meta_parameters"]

    assert registry["asum_clipping"]["mode"] == "add"
    assert registry["asum_clipping"]["validation"]["config_value_required"] is False
    assert registry["asum_clipping"]["validation"]["emitted_value_must_match"] is False

    assert registry["latitude"]["mode"] == "set"
    assert registry["latitude"]["validation"]["config_value_required"] is True
    assert registry["latitude"]["validation"]["emitted_value_must_match"] is True


def test_validate_metadata_allows_model_parameter_scope_mismatch():
    simtel_validate_metadata.validate_metadata(["metaparam telescope add array_triggers"])


def test_validate_metadata_rejects_generated_scope_mismatch():
    with pytest.raises(ValueError, match=r"scope mismatch for random_seed"):
        simtel_validate_metadata.validate_metadata(["metaparam telescope add random_seed"])
