"""Tests for differential integration output selection and comparison."""

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("differential_outputs.py")
MODULE_SPEC = importlib.util.spec_from_file_location("differential_outputs", MODULE_PATH)
differential_outputs = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(differential_outputs)


def test_selected_configs_uses_differential_test_flag(tmp_path):
    config_dir = tmp_path / "tests/integration_tests/config"
    config_dir.mkdir(parents=True)
    (config_dir / "selected.yml").write_text(
        "applications:\n- application: simtools-example\n  differential_test: true\n"
    )
    (config_dir / "ordinary.yml").write_text("applications:\n- application: simtools-example\n")

    assert differential_outputs._selected_configs(tmp_path) == ["selected.yml"]


def test_compare_trees_distinguishes_difference_and_comparison_error(tmp_path):
    base = tmp_path / "base"
    head = tmp_path / "head"
    base.mkdir()
    head.mkdir()
    (base / "result.yml").write_text("value: 1\n")
    (head / "result.yml").write_text("value: 2\n")

    assert differential_outputs.compare_trees(base, head, tmp_path / "diff.md") == 1

    (base / "result.yml").write_text("invalid: [\n")
    assert differential_outputs.compare_trees(base, head, tmp_path / "error.md") == 2
    assert "comparison failed" in (tmp_path / "error.md").read_text()


def test_normalization_preserves_paths_and_content_changes():
    assert differential_outputs.normalize_text("output=/tmp/base/result.txt") != (
        differential_outputs.normalize_text("output=/tmp/head/result.txt")
    )


def test_run_integration_rejects_empty_selection(tmp_path):
    (tmp_path / "tests/integration_tests/config").mkdir(parents=True)

    assert differential_outputs.run_integration(tmp_path, tmp_path / "outputs", "7.0.0") == 2
