"""Tests for the repository validation application."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from simtools.applications import validate_repository_consistency


def _write_metadata(directory, filename, product_id):
    """Write minimal product metadata for a test product."""
    metadata_file = directory / f"{filename}.meta.yml"
    metadata_file.write_text(
        f"cta:\n  product:\n    filename: {filename}\n    id: {product_id}\n",
        encoding="utf-8",
    )


def test_validates_product_references_and_allows_identical_product_ids(tmp_path):
    first = tmp_path / "input" / "first"
    second = tmp_path / "input" / "second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    for directory in (first, second):
        (directory / "product.ecsv").write_text("same\n", encoding="utf-8")
        _write_metadata(directory, "product.ecsv", "shared")

    assert (
        validate_repository_consistency.validate_repository_consistency(
            tmp_path, [Path("input")], Path("input")
        )
        == []
    )


def test_rejects_one_product_id_for_different_contents(tmp_path):
    first = tmp_path / "input" / "first"
    second = tmp_path / "input" / "second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "product.ecsv").write_text("first\n", encoding="utf-8")
    (second / "product.ecsv").write_text("second\n", encoding="utf-8")
    _write_metadata(first, "product.ecsv", "shared")
    _write_metadata(second, "product.ecsv", "shared")

    errors = validate_repository_consistency.validate_repository_consistency(
        tmp_path, [Path("input")], Path("input")
    )

    assert "product ID shared identifies different file contents" in "\n".join(errors)


def test_rejects_duplicate_active_workflow_versions(tmp_path):
    first = tmp_path / "input" / "first"
    second = tmp_path / "input" / "second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    config = (
        "applications:\n"
        "  - application: simtools-submit-model-parameter-from-external\n"
        "    configuration:\n"
        "      instrument: LSTN-01\n"
        "      parameter: test\n"
        "      parameter_version: 1.0.0\n"
    )
    (first / "config.yml").write_text(config, encoding="utf-8")
    (second / "config.yml").write_text(config, encoding="utf-8")

    errors = validate_repository_consistency.validate_repository_consistency(
        tmp_path, [Path("input")], Path("input")
    )

    assert "active workflows share" in "\n".join(errors)


def test_skip_marker_requires_single_line_reason(tmp_path):
    workflow = tmp_path / "input" / "workflow"
    workflow.mkdir(parents=True)
    (workflow / "config.yml").write_text("applications: []\n", encoding="utf-8")
    (workflow / "SKIP_WORKFLOW_CI").write_text("first\nsecond\n", encoding="utf-8")

    errors = validate_repository_consistency.validate_repository_consistency(
        tmp_path, [Path("input")], Path("input")
    )

    assert "provide a single-line reason" in "\n".join(errors)


def test_requires_git_tracked_products(monkeypatch, tmp_path):
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    (input_directory / "product.ecsv").write_text("data\n", encoding="utf-8")
    _write_metadata(input_directory, "product.ecsv", "product")
    repository = Mock(index=[])
    monkeypatch.setattr(
        validate_repository_consistency.pygit2, "Repository", Mock(return_value=repository)
    )

    errors = validate_repository_consistency.validate_repository_consistency(
        tmp_path, [Path("input")], Path("input"), require_git_tracking=True
    )

    assert "product file is missing or untracked" in "\n".join(errors)
    validate_repository_consistency.pygit2.Repository.assert_called_once_with(
        str(tmp_path.resolve())
    )


@patch.object(validate_repository_consistency, "validate_repository_consistency", return_value=[])
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_logs_success(mock_start, mock_validate):
    logger = Mock()
    mock_start.return_value = SimpleNamespace(
        args={
            "repository": Path(),
            "metadata_roots": [Path("input")],
            "workflow_root": Path("input"),
            "require_git_tracking": False,
        },
        logger=logger,
    )

    validate_repository_consistency.main()

    mock_validate.assert_called_once_with(
        Path(), [Path("input")], Path("input"), require_git_tracking=False
    )
    logger.info.assert_called_once_with(
        "Repository metadata references and workflow invariants are valid"
    )
