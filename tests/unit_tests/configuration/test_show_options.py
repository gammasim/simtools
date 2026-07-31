"""Tests for shared show-options handling."""

import argparse

import pytest

import simtools.configuration.show_options as show_options


def test_resolve_show_options_primary():
    result = show_options.resolve_show_options({"show_options": "primary"})

    assert result.option_name == "primary"
    assert "gamma" in result.values


def test_resolve_show_options_rejects_unknown_option():
    with pytest.raises(ValueError, match="Unsupported option"):
        show_options.resolve_show_options({"show_options": "unknown"})


def test_resolve_show_options_requires_model_version_for_array_layout_name():
    with pytest.raises(ValueError, match="requires '--model_version'"):
        show_options.resolve_show_options({"show_options": "array_layout_name"})


def test_resolve_show_options_rejects_multiple_model_versions_for_array_layout_name():
    with pytest.raises(ValueError, match="exactly one '--model_version'"):
        show_options.resolve_show_options(
            {"show_options": "array_layout_name", "model_version": ["7.0.0", "7.1.0"]}
        )


def test_handle_show_options_formats_values(capsys):
    with pytest.raises(SystemExit) as exc:
        show_options.handle_show_options(
            {"show_options": "site"},
            argparse.ArgumentParser(),
        )

    assert exc.value.code == 0
    assert "Available values:" in capsys.readouterr().out
