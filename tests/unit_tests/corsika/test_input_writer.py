"""Tests for schema-driven CORSIKA input serialization."""

import pytest

from simtools.corsika.input_writer import format_parameter, format_value


def test_schema_declared_converter_preserves_corsika_format():
    assert format_parameter("corsika_first_interaction_height", 1.2345, "km") == [
        "123450.00",
        "0",
    ]
    assert format_parameter("corsika_iact_io_buffer", 800, "MB") == "800MB"


def test_standard_formats():
    assert format_value(3, "scalar") == [3]
    assert format_value([1, 2], "list") == [1, 2]
    assert format_value(True, "bool_flag") == ["T"]
    assert format_value([1, 2], "quantity:cm", "m") == ["100.0", "200.0"]


def test_unknown_format_and_converter_fail():
    with pytest.raises(ValueError, match="Unknown CORSIKA input format"):
        format_value(1, "unknown")
    with pytest.raises(ValueError, match="Unknown CORSIKA input converter"):
        format_value(1, "converter:unknown")


def test_format_requires_schema_declaration():
    with pytest.raises(ValueError, match="No CORSIKA input format is declared"):
        format_parameter("unknown_parameter", 1)
