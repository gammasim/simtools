import logging
from pathlib import Path

import pytest
import yaml

from simtools.data_model import schema_loader
from simtools.io import ascii_handler

DUMMY_FILE = "dummy_file.yml"


@pytest.fixture(autouse=True)
def clear_schema_loader_cache():
    """Clear shared schema state around every test."""
    schema_loader._load_schema_source.cache_clear()
    yield
    schema_loader._load_schema_source.cache_clear()


@pytest.mark.parametrize("schema_directory", ["missing", None])
def test_get_model_parameter_schema_files_raises_for_missing_schemas(
    schema_directory, tmp_test_directory
):
    directory = tmp_test_directory / schema_directory if schema_directory else tmp_test_directory

    with pytest.raises(FileNotFoundError, match=r"^No schema files found"):
        schema_loader.get_model_parameter_schema_files(directory)


def test_get_model_parameter_schema_files(tmp_test_directory):
    tmp_test_directory = Path(tmp_test_directory)
    schema_files = [
        tmp_test_directory / "second.schema.yml",
        tmp_test_directory / "first.schema.yml",
    ]
    for schema_file in schema_files:
        schema_file.write_text(
            yaml.safe_dump({"name": schema_file.stem, "schema_version": "1.0.0"}),
            encoding="utf-8",
        )

    parameters, files = schema_loader.get_model_parameter_schema_files(tmp_test_directory)

    assert parameters == ["first.schema", "second.schema"]
    assert files == sorted(schema_files)


@pytest.mark.parametrize(
    ("schema_file", "error"),
    [
        ("missing.schema.yml", "Schema file not found: missing.schema.yml"),
        (
            "https://example.com/schemas/missing.schema.yml",
            "Schema file not found: https://example.com/schemas/missing.schema.yml",
        ),
    ],
)
def test_load_schema_reports_missing_source(schema_file, error, monkeypatch, tmp_test_directory):
    monkeypatch.setattr(schema_loader, "SCHEMA_PATH", tmp_test_directory)

    def _raise_file_not_found(file_name):
        raise FileNotFoundError(file_name)

    monkeypatch.setattr(ascii_handler, "collect_data_from_file", _raise_file_not_found)

    with pytest.raises(FileNotFoundError, match=error):
        schema_loader.load_schema(schema_file)


@pytest.mark.parametrize(
    ("schema_version", "expected_version"),
    [("latest", "2.0.0"), ("1.0.0", "1.0.0")],
)
def test_get_schema_for_version_with_list(schema_version, expected_version):
    schemas = [
        {"schema_version": "2.0.0", "name": "latest"},
        {"schema_version": "1.0.0", "name": "old"},
    ]

    result = schema_loader.get_schema_for_version(schemas, DUMMY_FILE, schema_version)

    assert result["schema_version"] == expected_version


@pytest.mark.parametrize(
    ("schemas", "schema_version", "error"),
    [
        ([], "latest", "No schemas found"),
        ([{"schema_version": "1.0.0"}], "2.0.0", "Schema version 2.0.0 not found"),
        ({"schema_version": "1.0.0"}, None, "Schema version not given"),
    ],
)
def test_get_schema_for_version_raises(schemas, schema_version, error):
    with pytest.raises(ValueError, match=error):
        schema_loader.get_schema_for_version(schemas, DUMMY_FILE, schema_version)


def test_get_schema_for_version_warns_on_mismatch(caplog):
    schema = {"schema_version": "1.0.0", "name": "test"}

    with caplog.at_level(logging.WARNING):
        result = schema_loader.get_schema_for_version(schema, DUMMY_FILE, "2.0.0")

    assert result is schema
    assert "Schema version 2.0.0 does not match 1.0.0" in caplog.text
