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
    assert registry["latitude"]["mode"] == "set"


def test_validate_metadata_allows_model_parameter_scope_mismatch():
    simtel_validate_metadata.validate_metadata(["metaparam telescope add array_triggers"])


def test_validate_metadata_uses_mode_for_required_values():
    simtel_validate_metadata.validate_metadata(["metaparam telescope add asum_clipping"])

    with pytest.raises(ValueError, match=r"value missing for required key latitude"):
        simtel_validate_metadata.validate_metadata(["metaparam global set latitude"])


def test_validate_metadata_values_checks_known_keys_and_skips_unknown_keys():
    simtel_validate_metadata.validate_metadata_values(
        {"azimuth_angle": "180.0", "sim_telarray_runtime_key": "not described"}
    )

    with pytest.raises(ValueError, match=r"could not convert string to float"):
        simtel_validate_metadata.validate_metadata_values({"azimuth_angle": "not-a-number"})


def test_validate_metadata_values_skips_add_and_model_derived_keys():
    simtel_validate_metadata.validate_metadata_values(
        {
            "random_seed": "1745,290",
            "discriminator_output_amplitude": "not-a-number",
        }
    )


def test_validate_metadata_rejects_generated_scope_mismatch():
    with pytest.raises(ValueError, match=r"scope mismatch for random_seed"):
        simtel_validate_metadata.validate_metadata(["metaparam telescope add random_seed"])
