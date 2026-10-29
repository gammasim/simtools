"""Tests for the differential output comparison script."""

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.table import Table

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "integration_tests/differential_outputs.py"
SPEC = importlib.util.spec_from_file_location("differential_outputs", SCRIPT_PATH)
DIFFERENTIAL_OUTPUTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIFFERENTIAL_OUTPUTS)


def test_normalize_text_preserves_dates_hashes_and_ids():
    text = (
        "date=2025-02-01T12:30:40Z "
        "hash=1234567890abcdef1234567890abcdef12345678 "
        "id=123e4567-e89b-12d3-a456-426614174000"
    )

    normalized = DIFFERENTIAL_OUTPUTS.normalize_text(text)

    assert normalized == text


def test_normalize_generated_metadata_preserves_scientific_changes():
    def metadata(root, activity, energy):
        return {
            "created": activity,
            "energy": energy,
            "cta": {
                "activity": {"id": activity, "start": activity, "end": activity},
                "product": {"id": activity, "creation_time": activity},
                "context": {
                    "application_configuration": {
                        "arguments": {
                            "activity_id": activity,
                            "output_path": f"{root}/test-data1/app/output",
                        }
                    }
                },
            },
        }

    base = DIFFERENTIAL_OUTPUTS.normalize_value(metadata("/tmp/base", "one", 1))
    assert base == DIFFERENTIAL_OUTPUTS.normalize_value(metadata("/tmp/head", "two", 1))
    assert base != DIFFERENTIAL_OUTPUTS.normalize_value(metadata("/tmp/head", "two", 2))


def test_relative_files_excludes_generated_config(tmp_test_directory):
    root = Path(tmp_test_directory)
    (root / "tmp_config.yml").write_text("temporary: true\n")
    (root / "result.yml").write_text("value: 1\n")
    assert set(DIFFERENTIAL_OUTPUTS._relative_files(root)) == {Path("result.yml")}


def test_compare_table_uses_numeric_tolerance_and_units(tmp_test_directory):
    base = Path(tmp_test_directory) / "base.ecsv"
    head = Path(tmp_test_directory) / "head.ecsv"
    Table({"energy": [1.0]}, units=["TeV"]).write(base, format="ascii.ecsv")
    Table({"energy": [1.0 + 1e-13]}, units=["TeV"]).write(head, format="ascii.ecsv")

    assert DIFFERENTIAL_OUTPUTS._compare_tables(base, head) == []


@pytest.mark.parametrize(
    ("suffix", "base_text", "head_text", "expected"),
    [
        (".json", '{"value": 1}', '{"value": 1.1}', "JSON structures differ"),
        (".yaml", "value: one\n", "value: two\n", "YAML structures differ"),
        (".txt", "first\nsecond\n", "first\nchanged\n", "line 2:"),
    ],
)
def test_compare_structured_and_text_files(
    tmp_test_directory, suffix, base_text, head_text, expected
):
    base = Path(tmp_test_directory) / f"base{suffix}"
    head = Path(tmp_test_directory) / f"head{suffix}"
    base.write_text(base_text, encoding="utf-8")
    head.write_text(head_text, encoding="utf-8")

    assert expected in "\n".join(DIFFERENTIAL_OUTPUTS._compare_file(base, head))


def test_compare_hdf5_dataset_values(tmp_test_directory):
    base = Path(tmp_test_directory) / "base.hdf5"
    head = Path(tmp_test_directory) / "head.hdf5"
    for path, value in ((base, 1.0), (head, 2.0)):
        with h5py.File(path, "w") as output:
            output.create_dataset("values", data=np.array([value]))

    assert "values: numeric values differ" in DIFFERENTIAL_OUTPUTS._hdf5_differences(base, head)


def test_compare_hdf5_preserves_meaningful_json_metadata(tmp_test_directory):
    base = Path(tmp_test_directory) / "base.hdf5"
    head = Path(tmp_test_directory) / "head.hdf5"
    metadata = [
        json.dumps(
            {
                "activity_id": "123e4567-e89b-12d3-a456-426614174000",
                "path": "/tmp/base/output.txt",
                "time": "2025-02-01T12:30:40Z",
            }
        ),
        json.dumps(
            {
                "activity_id": "223e4567-e89b-12d3-a456-426614174000",
                "path": "/tmp/head/output.txt",
                "time": "2026-02-01T12:30:40Z",
            }
        ),
    ]
    for path, value in zip((base, head), metadata):
        with h5py.File(path, "w") as output:
            output.create_dataset("METADATA", data=np.array([value.encode()]))

    assert "METADATA: values differ" in DIFFERENTIAL_OUTPUTS._hdf5_differences(base, head)


def test_compare_report_exit_code_and_binary_informational(tmp_test_directory):
    root = Path(tmp_test_directory)
    base, head = root / "base", root / "head"
    base.mkdir()
    head.mkdir()
    (base / "result.txt").write_text("base\n", encoding="utf-8")
    (head / "result.txt").write_text("head\n", encoding="utf-8")
    (base / "image.png").write_bytes(b"a")
    (head / "image.png").write_bytes(b"different")
    report = root / "diff_report.md"

    result = DIFFERENTIAL_OUTPUTS.compare_trees(base, head, report)

    assert result == 1
    report_text = report.read_text(encoding="utf-8")
    assert "result.txt" in report_text
    assert "informational" in report_text


def test_compare_equal_trees_succeeds(tmp_test_directory):
    root = Path(tmp_test_directory)
    base, head = root / "base", root / "head"
    base.mkdir()
    head.mkdir()
    (base / "result.json").write_text('{"value": 1}\n', encoding="utf-8")
    (head / "result.json").write_text('{"value": 1.0000000000001}\n', encoding="utf-8")

    assert DIFFERENTIAL_OUTPUTS.compare_trees(base, head, root / "diff_report.md") == 0


def test_simtools_software_version_is_normalized():
    base = {"software": {"name": "simtools", "version": "0.38.4.dev90+g52213ce96"}}
    head = {"software": {"name": "simtools", "version": "0.38.4.dev91+g1234567"}}
    assert DIFFERENTIAL_OUTPUTS.normalize_value(base) == DIFFERENTIAL_OUTPUTS.normalize_value(head)
    model = {"software": {"name": "sim_telarray", "version": "2024.1"}}
    assert DIFFERENTIAL_OUTPUTS.normalize_value(model) == model


def test_tables_with_different_row_counts(tmp_test_directory):
    base_file = Path(tmp_test_directory) / "base.ecsv"
    head_file = Path(tmp_test_directory) / "head.ecsv"
    Table({"value": [1]}).write(base_file, overwrite=True)
    Table({"value": [1, 2]}).write(head_file, overwrite=True)
    assert DIFFERENTIAL_OUTPUTS._compare_tables(base_file, head_file) == ["rows: 1 != 2"]


def test_selected_configs_uses_differential_test_flag(tmp_test_directory):
    config_dir = Path(tmp_test_directory) / "tests/integration_tests/config"
    config_dir.mkdir(parents=True)
    (config_dir / "selected.yml").write_text(
        "applications:\n- application: simtools-example\n  differential_test: true\n"
    )
    (config_dir / "ordinary.yml").write_text("applications:\n- application: simtools-example\n")

    assert DIFFERENTIAL_OUTPUTS._selected_configs(tmp_test_directory) == ["selected.yml"]


def test_compare_trees_distinguishes_difference_and_comparison_error(tmp_test_directory):
    base = Path(tmp_test_directory) / "base"
    head = Path(tmp_test_directory) / "head"
    base.mkdir()
    head.mkdir()
    (base / "result.yml").write_text("value: 1\n")
    (head / "result.yml").write_text("value: 2\n")

    assert DIFFERENTIAL_OUTPUTS.compare_trees(base, head, Path(tmp_test_directory) / "diff.md") == 1

    (base / "result.yml").write_text("invalid: [\n")
    assert (
        DIFFERENTIAL_OUTPUTS.compare_trees(base, head, Path(tmp_test_directory) / "error.md") == 2
    )
    assert "comparison failed" in (Path(tmp_test_directory) / "error.md").read_text()


def test_normalization_preserves_paths_and_content_changes():
    assert DIFFERENTIAL_OUTPUTS.normalize_text("output=/tmp/base/result.txt") != (
        DIFFERENTIAL_OUTPUTS.normalize_text("output=/tmp/head/result.txt")
    )


def test_run_integration_rejects_empty_selection(tmp_test_directory):
    (Path(tmp_test_directory) / "tests/integration_tests/config").mkdir(parents=True)

    assert (
        DIFFERENTIAL_OUTPUTS.run_integration(
            tmp_test_directory, Path(tmp_test_directory) / "outputs", "7.0.0"
        )
        == 2
    )
