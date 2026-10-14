"""Compatibility decisions use exact model identities, never directory names."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from simtools.model_repository import compatibility as compat

DEFAULT_PATTERN = "default"
MODELS_NAME = "models"
NORMALIZED_VERSION = "0.17.1"
CHANGED = "changed"
ENCODING = "utf-8"
PARAMETER_FILE = "parameter.json"
NEW_VERSION = "v0.18.0"
NEW_RANGE = "==0.18.0"

MODEL = compat.MODEL_KEY
DEFAULT = "default-ref"
VERSION = "v0.17.1"
RANGE = ">=0.17.1,<0.17.2"


@pytest.fixture
def models(tmp_test_directory):
    root = Path(tmp_test_directory) / MODELS_NAME
    data = root / compat.MODELS_DIRECTORY
    data.mkdir(parents=True)
    (data / PARAMETER_FILE).write_text('{"value": 1}', encoding=ENCODING)
    for args in (
        ("init", "-q"),
        ("add", "."),
        ("-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", MODELS_NAME),
        ("tag", VERSION),
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return root


def catalog(default=VERSION, allowed=RANGE):
    return {MODEL: {DEFAULT: default, compat.RANGE_KEY: allowed}}


@pytest.mark.parametrize("allowed", ["nonsense", ">=0.18.0", "==0.17.1rc1"])
def test_contract_rejects_invalid_or_inconsistent_range(allowed):
    with pytest.raises(ValueError, match=DEFAULT_PATTERN):
        compat._contract(catalog(allowed=allowed))


def test_legacy_contract_uses_exact_default():
    contract = compat._contract({compat.LEGACY_MODEL_KEY: {"default-tag": VERSION}})
    assert contract[compat.ALLOWED_KEY] == "==0.17.1"
    assert contract[compat.POLICY_KEY] == compat.LEGACY_POLICY


@pytest.mark.parametrize(DEFAULT_PATTERN, ["main", "v0.17.1rc1", None])
def test_unknown_or_prerelease_defaults_are_not_production_contracts(default):
    with pytest.raises(ValueError, match=DEFAULT_PATTERN):
        compat._contract(catalog(default=default))


def test_clean_tagged_checkout_and_git_source_have_same_digest(models):
    identity = compat._identity(models)
    assert identity[compat.VERSION_KEY] == NORMALIZED_VERSION
    assert identity == compat._identity(models, VERSION)
    assert compat._validate(models, catalog())[compat.STATUS_KEY] == "compatible"


def test_incompatible_dependency_is_actionable(models):
    with pytest.raises(ValueError, match="No production jobs were submitted"):
        compat._validate(models, catalog(default=NEW_VERSION, allowed=NEW_RANGE))


def test_reject_dirty_models_but_fixed_git_revision_remains_valid(models):
    (models / compat.MODELS_DIRECTORY / PARAMETER_FILE).write_text(CHANGED, encoding=ENCODING)
    with pytest.raises(ValueError, match="local modifications"):
        compat._identity(models)
    assert compat._identity(models, VERSION)[compat.VERSION_KEY] == NORMALIZED_VERSION
    result = compat._validate(models, catalog(), development=True)
    assert result[compat.STATUS_KEY] == compat.DEVELOPMENT_STATUS


def test_ambiguous_tags_are_not_silently_selected(models):
    compat._git(models, "tag", "v0.17.2")
    with pytest.raises(ValueError, match="unambiguous"):
        compat._identity(models)


def test_verified_model_export_and_content_drift(models):
    identity = compat._identity(models)
    shutil.rmtree(models / ".git")
    manifest = models / compat.RELEASE_MANIFEST
    manifest.write_text(json.dumps(identity), encoding=ENCODING)
    assert compat._identity(models) == identity
    (models / compat.MODELS_DIRECTORY / "extra").write_text(CHANGED, encoding=ENCODING)
    with pytest.raises(ValueError, match="digest is invalid"):
        compat._identity(models)


def test_unknown_export_and_symbolic_link_rejected(models):
    empty = models.parent / "unknown"
    empty.mkdir()
    with pytest.raises(ValueError, match="identity unknown"):
        compat._identity(empty)
    (models / compat.MODELS_DIRECTORY / "link").symlink_to(models / ".git" / "HEAD")
    with pytest.raises(ValueError, match="symbolic links"):
        compat._tree_digest(models)


def test_contract_requires_explicit_dependency_and_constraint():
    with pytest.raises(ValueError, match="no model dependency"):
        compat._contract({})
    with pytest.raises(ValueError, match="nonempty"):
        compat._contract(catalog(allowed=""))


def test_missing_model_directory_and_unknown_development_release(models):
    with pytest.raises(ValueError, match="missing"):
        compat._tree_digest(models.parent)
    assert (
        compat._validate(models, catalog(default=NEW_VERSION, allowed=NEW_RANGE), development=True)[
            compat.STATUS_KEY
        ]
        == compat.DEVELOPMENT_STATUS
    )


def test_ignored_model_changes_are_not_release_contents(models):
    (models / ".gitignore").write_text("ignored\n", encoding=ENCODING)
    (models / compat.MODELS_DIRECTORY / "ignored").write_text(CHANGED, encoding=ENCODING)
    with pytest.raises(ValueError, match="contents differ"):
        compat._identity(models)
