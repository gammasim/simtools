#!/usr/bin/env python3

import importlib.util
import urllib.error
from pathlib import Path

import pytest


def _load_generate_test_resources_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "resources_generation" / "generate_test_resources.py"
    )
    spec = importlib.util.spec_from_file_location("generate_test_resources", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_test_resources = _load_generate_test_resources_module()


def test_copy_manual_fixtures(tmp_test_directory):
    source_dir = Path(tmp_test_directory) / "manual_fixtures"
    source_dir.mkdir()
    (source_dir / "fixture.txt").write_text("manual fixture\n", encoding="utf-8")

    nested_dir = source_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "entry.txt").write_text("nested fixture\n", encoding="utf-8")

    config_file = Path(tmp_test_directory) / "manual_fixture.yml"
    config_file.write_text(
        """
files:
  - source_path: fixture.txt
    description: simple file fixture
    target_path: copied/fixture.txt
  - source_path: nested
    description: nested directory fixture
    target_path: copied/nested
""".strip()
        + "\n",
        encoding="utf-8",
    )

    resources_dir = Path(tmp_test_directory) / "resources"
    generate_test_resources.copy_manual_fixtures(
        config_file=config_file,
        source_dir=source_dir,
        target_dir=resources_dir,
    )

    assert (resources_dir / "copied" / "fixture.txt").exists()
    assert (resources_dir / "copied" / "nested" / "entry.txt").exists()


def test_copy_manual_fixtures_missing_required_field(tmp_test_directory):
    source_dir = Path(tmp_test_directory) / "manual_fixtures"
    source_dir.mkdir()

    config_file = Path(tmp_test_directory) / "manual_fixture.yml"
    config_file.write_text(
        """
files:
  - source_path: fixture.txt
    description: simple file fixture
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required keys: target_path"):
        generate_test_resources.copy_manual_fixtures(
            config_file=config_file,
            source_dir=source_dir,
        )


def test_copy_manual_fixtures_missing_source_path(tmp_test_directory):
    source_dir = Path(tmp_test_directory) / "manual_fixtures"
    source_dir.mkdir()

    config_file = Path(tmp_test_directory) / "manual_fixture.yml"
    config_file.write_text(
        """
files:
  - source_path: missing.txt
    description: missing fixture
    target_path: copied/missing.txt
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Manual fixture 'missing fixture' does not exist"):
        generate_test_resources.copy_manual_fixtures(
            config_file=config_file,
            source_dir=source_dir,
        )


def test_download_files(tmp_test_directory, monkeypatch):
    config_file = Path(tmp_test_directory) / "download_files.yml"
    config_file.write_text(
        """
files:
  - url: https://example.org/test.csv
    description: test file
    target_path: folder/test.csv
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def _fake_urlretrieve(url, destination):
        assert url == "https://example.org/test.csv"
        Path(destination).write_text("value\n", encoding="utf-8")
        return destination, None

    monkeypatch.setattr(generate_test_resources.urllib.request, "urlretrieve", _fake_urlretrieve)

    resources_dir = Path(tmp_test_directory) / "resources"
    generate_test_resources.download_files(config_file=config_file, target_dir=resources_dir)

    assert (resources_dir / "folder" / "test.csv").exists()


def test_download_files_missing_required_field(tmp_test_directory):
    config_file = Path(tmp_test_directory) / "download_files.yml"
    config_file.write_text(
        """
files:
  - url: https://example.org/test.csv
    description: test file
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required keys: target_path"):
        generate_test_resources.download_files(config_file=config_file)


def test_download_files_download_failure(tmp_test_directory, monkeypatch):
    config_file = Path(tmp_test_directory) / "download_files.yml"
    config_file.write_text(
        """
files:
  - url: https://example.org/test.csv
    description: test file
    target_path: folder/test.csv
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def _fail_urlretrieve(url, destination):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(generate_test_resources.urllib.request, "urlretrieve", _fail_urlretrieve)

    with pytest.raises(FileNotFoundError, match="Failed to download 'test file'"):
        generate_test_resources.download_files(config_file=config_file)
