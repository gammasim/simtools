"""Model releases must satisfy the runtime's catalog constraint."""

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
        models._contract({models.MODEL_KEY: {DEFAULT_REF: default}})


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
