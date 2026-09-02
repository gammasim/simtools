"""Tests for production-table file handling."""

import json
from pathlib import Path

from simtools.model_repository.files import get_production_table_files, read_production_tables


def test_read_production_tables_aggregates_parameters_and_removes_deprecated(tmp_test_directory):
    """A production JSON file is indexed by collection and cleaned up."""
    model_path = Path(tmp_test_directory) / "models" / "simulation-models" / "productions" / "1.0.0"
    model_path.mkdir(parents=True)
    production_file = model_path / "LSTN-01.json"
    production_file.write_text(
        json.dumps(
            {
                "parameters": {"LSTN-01": {"camera_body_diameter": "1.0.0", "obsolete": "1.0.0"}},
                "design_model": {"LSTN-01": "LSTN-design"},
                "deprecated_parameters": ["obsolete"],
            }
        ),
        encoding="utf-8",
    )
    (model_path / "LSTN-02.json").write_text(
        json.dumps({"parameters": {"LSTN-02": {"camera_body_diameter": "1.0.0"}}}),
        encoding="utf-8",
    )

    assert get_production_table_files(model_path) == [
        ("1.0.0", production_file),
        ("1.0.0", model_path / "LSTN-02.json"),
    ]
    table = read_production_tables(model_path, collection_name="telescopes")["telescopes"]

    assert table["collection"] == "telescopes"
    assert table["model_version"] == "1.0.0"
    assert table["parameters"] == {
        "LSTN-01": {"camera_body_diameter": "1.0.0"},
        "LSTN-02": {"camera_body_diameter": "1.0.0"},
    }
    assert table["design_model"] == {"LSTN-01": "LSTN-design"}


def test_get_production_table_files_includes_patch_history(tmp_test_directory):
    """Patch metadata adds production files from each referenced version."""
    productions = Path(tmp_test_directory) / "models" / "simulation-models" / "productions"
    model_path = productions / "1.1.0"
    model_path.mkdir(parents=True)
    (model_path / "info.yml").write_text(
        "model_update: patch_update\nmodel_version_history:\n  - 1.0.0\n", encoding="utf-8"
    )
    (model_path / "LSTN-01.json").write_text("{}", encoding="utf-8")
    previous_path = productions / "1.0.0"
    previous_path.mkdir()
    (previous_path / "LSTN-design.json").write_text("{}", encoding="utf-8")

    files = get_production_table_files(model_path)

    assert files == [
        ("1.0.0", previous_path / "LSTN-design.json"),
        ("1.1.0", model_path / "LSTN-01.json"),
    ]
