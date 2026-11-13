"""Model releases must satisfy the runtime's catalog constraint."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from simtools.model_repository import compatibility as models

CONSTRAINT = "constraint"
DEFAULT_REF = "default-ref"
INVALID_CONTRACT = "Invalid|nonempty|not a supported stable release"

LEGACY_DEFAULT = "v0.17.1"
CATALOG = {models.LEGACY_MODEL_KEY: {"default-tag": LEGACY_DEFAULT}}
RANGE = ">=0.17.1,<0.18.0"


def test_release_in_repository_with_different_owner(monkeypatch, tmp_test_directory):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    source = str(Path(tmp_test_directory).resolve())
    git = ["git", "-C", source]
    subprocess.run([*git, "init"], check=True, capture_output=True)
    subprocess.run(
        [
            *git,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "model fixture",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([*git, "tag", LEGACY_DEFAULT], check=True, capture_output=True)
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    untrusted = subprocess.run([*git, "tag", "--points-at", "HEAD"], capture_output=True)
    assert untrusted.returncode != 0
    assert b"dubious ownership" in untrusted.stderr

    result = models.validate_release(tmp_test_directory, CATALOG)
    assert result[models.STATUS_KEY] == "compatible"
    assert result[models.VERSION_KEY] == "0.17.1"


@pytest.mark.parametrize(("tag", "accepted"), [(LEGACY_DEFAULT, True), ("v0.17.2", False)])
def test_historical_release_constraint(monkeypatch, tmp_test_directory, tag, accepted):
    monkeypatch.setattr(models.subprocess, "run", Mock(return_value=SimpleNamespace(stdout=tag)))
    if accepted:
        result = models.validate_release(tmp_test_directory, CATALOG)
        assert result[models.VERSION_KEY] == "0.17.1"
        assert result[models.STATUS_KEY] == "compatible"
    else:
        with pytest.raises(ValueError, match=r"Allowed ==0\.17\.1"):
            models.validate_release(tmp_test_directory, CATALOG)


@pytest.mark.parametrize("default", ["main", "v0.18.0rc1", "v0.18.0.dev1"])
def test_unreleased_default(default):
    with pytest.raises(ValueError, match=INVALID_CONTRACT):
        models.get_contract({models.MODEL_KEY: {DEFAULT_REF: default}})


@pytest.mark.parametrize("tags", ["main", "v0.17.1\nv0.17.2", ""])
@pytest.mark.parametrize("development", [False, True])
def test_unknown_version(monkeypatch, tmp_test_directory, tags, development):
    monkeypatch.setattr(models.subprocess, "run", Mock(return_value=SimpleNamespace(stdout=tags)))
    if development:
        result = models.validate_release(tmp_test_directory, CATALOG, development=True)
        assert result[models.STATUS_KEY] == models.DEVELOPMENT_STATUS
    else:
        with pytest.raises(ValueError, match="Tags at selected revision") as error:
            models.validate_release(tmp_test_directory, CATALOG)
        found_tags = ", ".join(tags.splitlines()) if tags else "none"
        assert f"Tags at selected revision: {found_tags}." in str(error.value)
        assert "Required: exactly one stable release tag. Allowed ==0.17.1" in str(error.value)
