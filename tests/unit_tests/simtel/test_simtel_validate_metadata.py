#!/usr/bin/python3

import pytest

from simtools.simtel import simtel_validate_metadata


def test_get_meta_parameter_registry_uses_generated_and_model_parameter_sources():
    registry = simtel_validate_metadata.get_meta_parameter_registry()["meta_parameters"]

    assert registry["simtools_version"]["source_type"] == "generated"
    assert registry["random_mono_prob"]["source_type"] == "model_parameter"


def test_get_meta_parameter_registry_filters_by_source_type():
    registry = simtel_validate_metadata.get_meta_parameter_registry(source_type="generated")[
        "meta_parameters"
    ]

    assert "simtools_version" in registry
    assert "random_mono_prob" not in registry
    assert {definition["source_type"] for definition in registry.values()} == {"generated"}


def test_get_meta_parameter_registry_rejects_invalid_source_type():
    with pytest.raises(ValueError, match=r"Unsupported source type"):
        simtel_validate_metadata.get_meta_parameter_registry(source_type="unknown")


def test_validate_metadata_allows_model_parameter_scope_mismatch():
    assert (
        simtel_validate_metadata.validate_metadata(["metaparam telescope add array_triggers"])
        is None
    )


def test_validate_metadata_rejects_unknown_key_and_mode_mismatch():
    with pytest.raises(KeyError, match=r"unknown_key"):
        simtel_validate_metadata.validate_metadata(["metaparam global set unknown_key = value"])

    with pytest.raises(ValueError, match=r"mode mismatch for primary"):
        simtel_validate_metadata.validate_metadata(["metaparam global add primary"])


def test_validate_metadata_uses_mode_for_required_values():
    simtel_validate_metadata.validate_metadata(["metaparam telescope add asum_clipping"])
    simtel_validate_metadata.validate_metadata(["config_release = test-release"])
    simtel_validate_metadata.validate_metadata(["metaparam global set primary = gamma"])

    with pytest.raises(ValueError, match=r"value missing for required key latitude"):
        simtel_validate_metadata.validate_metadata(["metaparam global set latitude"])


def test_validate_metadata_values_checks_known_keys_and_skips_unknown_keys():
    simtel_validate_metadata.validate_metadata_values(
        {"azimuth_angle": "180.0", "sim_telarray_runtime_key": "not described"}
    )

    with pytest.raises(ValueError, match=r"could not convert string to float"):
        simtel_validate_metadata.validate_metadata_values({"azimuth_angle": "not-a-number"})


def test_validate_metadata_values_accepts_legacy_generated_names():
    simtel_validate_metadata.validate_metadata_values({"simtools_simtel_version": "v2025-11-30-rc"})


def test_validate_metadata_values_skips_add_and_model_derived_keys():
    assert (
        simtel_validate_metadata.validate_metadata_values(
            {
                "random_seed": "1745,290",
                "discriminator_output_amplitude": "not-a-number",
            }
        )
        is None
    )


def test_parse_metadata_line_rejects_malformed_lines():
    for line in (
        "metaparam invalid set primary = gamma",
        "metaparam global invalid primary = gamma",
        "metaparam global set invalid name = gamma",
        "not metadata",
    ):
        with pytest.raises(ValueError, match=r"Unsupported sim_telarray metadata line"):
            simtel_validate_metadata._parse_metadata_line(line)


def test_validate_metadata_rejects_generated_scope_mismatch():
    with pytest.raises(ValueError, match=r"scope mismatch for random_seed"):
        simtel_validate_metadata.validate_metadata(["metaparam telescope add random_seed"])


@pytest.mark.parametrize(
    ("model_schema", "expected"),
    [
        ({"data": {"type": "string"}}, {"kind": "scalar", "data_type": "string"}),
        ({"data": []}, {"kind": "scalar", "data_type": "string"}),
        (
            {"data": [{"type": "float64"}, {"type": "double"}]},
            {"kind": "fixed_numeric_tuple", "item_type": "number", "length": 2},
        ),
        ({"data": [{"type": "file", "default": "test.dat"}]}, {"kind": "file_name"}),
        (
            {"data": [{"type": "file", "default": None}]},
            {"kind": "file_name", "allow_none_literal": True},
        ),
        ({"data": [{"type": "uint64"}]}, {"kind": "scalar", "data_type": "integer"}),
        ({"data": [{"type": "float32"}]}, {"kind": "scalar", "data_type": "number"}),
        ({"data": [{"type": "boolean"}]}, {"kind": "scalar", "data_type": "boolean"}),
    ],
)
def test_derive_value_schema(model_schema, expected):
    assert simtel_validate_metadata._derive_value_schema(model_schema) == expected


def test_validate_metadata_value_schema_kinds():
    assert (
        simtel_validate_metadata._validate_metadata_value(
            "test", "none", {"kind": "file_name", "allow_none_literal": True}
        )
        is None
    )
    assert (
        simtel_validate_metadata._validate_metadata_value(
            "test", "1.0 2.0", {"kind": "fixed_numeric_tuple", "length": 2}
        )
        is None
    )
    assert (
        simtel_validate_metadata._validate_metadata_value(
            "test", "all: 1.0", {"kind": "sim_telarray_key_value_string", "regex": r"all: \d+\.\d+"}
        )
        is None
    )
    assert (
        simtel_validate_metadata._validate_metadata_value(
            "test", "True", {"kind": "scalar", "data_type": "boolean"}
        )
        is None
    )
    assert (
        simtel_validate_metadata._validate_metadata_value(
            "test", "10", {"kind": "scalar", "data_type": "integer"}
        )
        is None
    )


@pytest.mark.parametrize(
    ("value_schema", "value", "match"),
    [
        ({"kind": "file_name"}, "", r"Empty file-like"),
        ({"kind": "fixed_numeric_tuple", "length": 2}, "1.0", r"tuple length mismatch"),
        (
            {"kind": "sim_telarray_key_value_string", "regex": r"all: \d+\.\d+"},
            "bad",
            r"does not match regex",
        ),
        ({"kind": "scalar", "data_type": "string"}, "", r"Empty string"),
        ({"kind": "scalar", "data_type": "boolean"}, "maybe", r"Invalid boolean"),
        ({"kind": "scalar", "data_type": "object"}, "value", r"Unsupported scalar"),
        ({"kind": "unknown"}, "value", r"Unsupported value schema kind"),
    ],
)
def test_validate_metadata_value_rejects_invalid_values(value_schema, value, match):
    with pytest.raises(ValueError, match=match):
        simtel_validate_metadata._validate_metadata_value("test", value, value_schema)
