"""Tests for explicit output paths."""

import pytest

from simtools.io.output_paths import OutputPaths


def test_output_paths_create_directories_and_files(tmp_test_directory):
    """Output directories and file parents are created on demand."""
    paths = OutputPaths(tmp_test_directory)

    directory = paths.get_output_directory("run")
    output_file = paths.get_output_file("result.txt", sub_dir="run")

    assert directory.is_dir()
    assert output_file.parent == directory


def test_output_paths_requires_configured_label():
    """An unknown output label identifies the missing configuration."""
    paths = OutputPaths()

    with pytest.raises(KeyError, match="Output path label 'missing' not found"):
        paths.get_output_directory(output_path_label="missing")
