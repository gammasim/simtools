"""Tests for resolving runtime release defaults from configured repositories."""

import json
from unittest.mock import MagicMock, Mock

import pytest
import yaml

from simtools.runners import runtime_releases


@pytest.fixture
def catalog():
    return {
        "model-repository": {"repository-url": "https://gitlab.example.org/team/models.git"},
        "images": {"registry": "registry.example.org/example"},
    }


@pytest.fixture
def release_catalog():
    return {
        "schema_version": "0.9.0",
        "images": {"registry": "registry.example.org/example", "tag-suffix": "-gl1"},
        "cpu-variants": ["generic"],
        "production-combinations": [{"corsika": "v7.8010", "sim-telarray": "v1"}],
        "corsika": [{"source-ref": "v7.8010"}],
        "sim-telarray": [{"source-ref": "v1"}],
    }


@pytest.fixture
def repository_metadata(monkeypatch):
    monkeypatch.setattr(
        runtime_releases,
        "metadata",
        Mock(
            return_value=MagicMock(
                get_all=Mock(return_value=["repository, https://github.com/example/tools"])
            )
        ),
    )


def test_latest_image_uses_catalog_registry_and_release_matrix(
    monkeypatch, catalog, release_catalog, repository_metadata
):
    digest = "sha256:" + "a" * 64
    resources = [
        json.dumps({"tag_name": "v1.2.3"}),
        yaml.safe_dump(release_catalog),
        json.dumps({"token": "test-token"}),
    ]
    reader = Mock(side_effect=resources)
    monkeypatch.setattr(runtime_releases, "_read_url", reader)
    response = MagicMock()
    response.__enter__.return_value.headers = {
        "Docker-Content-Digest": digest,
        "Content-Type": "application/vnd.oci.image.index.v1+json",
    }
    opener = Mock(return_value=response)
    monkeypatch.setattr(runtime_releases, "urlopen", opener)
    assert (
        runtime_releases.latest_simtools_image(catalog)
        == f"registry.example.org/example/simtools-prod@{digest}"
    )
    assert (
        reader.call_args_list[0].args[0]
        == "https://api.github.com/repos/example/tools/releases/latest"
    )
    assert reader.call_args_list[2].args[0].startswith("https://registry.example.org/token?")
    assert opener.call_args.args[0].full_url == (
        "https://registry.example.org/v2/example/simtools-prod/manifests/v1.2.3-v78010-v1-generic"
    )


def test_read_url_reads_public_response(monkeypatch):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b"content"
    monkeypatch.setattr(runtime_releases, "urlopen", Mock(return_value=response))
    assert runtime_releases._read_url("https://example.org/release") == b"content"


@pytest.mark.parametrize(
    ("media_type", "digest", "message"),
    [
        ("application/vnd.oci.image.manifest.v1+json", "sha256:" + "a" * 64, "multi-platform"),
        ("application/vnd.oci.image.index.v1+json", "invalid", "image digest"),
    ],
)
def test_latest_image_rejects_invalid_registry_response(
    monkeypatch, catalog, release_catalog, repository_metadata, media_type, digest, message
):
    resources = [
        json.dumps({"tag_name": "v1.2.3"}),
        yaml.safe_dump(release_catalog),
        '{"token":"x"}',
    ]
    monkeypatch.setattr(runtime_releases, "_read_url", Mock(side_effect=resources))
    response = MagicMock()
    response.__enter__.return_value.headers = {
        "Content-Type": media_type,
        "Docker-Content-Digest": digest,
    }
    monkeypatch.setattr(runtime_releases, "urlopen", Mock(return_value=response))
    with pytest.raises(ValueError, match=message):
        runtime_releases.latest_simtools_image(catalog)


def test_latest_image_rejects_nonrelease_tag(monkeypatch, catalog, repository_metadata):
    monkeypatch.setattr(
        runtime_releases, "_read_url", Mock(return_value='{"tag_name":"v1.0.0rc1"}')
    )
    with pytest.raises(ValueError, match="Unexpected simtools release tag"):
        runtime_releases.latest_simtools_image(catalog)


def test_latest_image_requires_generic_combination(
    monkeypatch, catalog, release_catalog, repository_metadata
):
    release_catalog["cpu-variants"] = ["avx2"]
    monkeypatch.setattr(
        runtime_releases,
        "_read_url",
        Mock(side_effect=[json.dumps({"tag_name": "v1.2.3"}), yaml.safe_dump(release_catalog)]),
    )
    with pytest.raises(ValueError, match="does not define a generic production image"):
        runtime_releases.latest_simtools_image(catalog)


def test_latest_model_release_uses_configured_repository(monkeypatch, catalog):
    releases = [
        {"tag_name": "v2.0.0", "upcoming_release": True, "commit": {"id": "a" * 40}},
        {"tag_name": "v2.0.0rc1", "commit": {"id": "a" * 40}},
        {"tag_name": "v1.10.0", "commit": {"id": "b" * 40}},
    ]
    reader = Mock(return_value=json.dumps(releases))
    monkeypatch.setattr(runtime_releases, "_read_url", reader)
    assert runtime_releases.latest_models_commit(catalog) == "b" * 40
    assert reader.call_args.args[0] == (
        "https://gitlab.example.org/api/v4/projects/team%2Fmodels/releases?per_page=100"
    )


def test_latest_models_fails_without_stable_release(monkeypatch, catalog):
    monkeypatch.setattr(runtime_releases, "_read_url", Mock(return_value="[]"))
    with pytest.raises(ValueError, match="No stable simulation-models release"):
        runtime_releases.latest_models_commit(catalog)


def test_latest_models_rejects_invalid_commit(monkeypatch, catalog):
    releases = [{"tag_name": "v1.0.0", "commit": {"id": "not-a-commit"}}]
    monkeypatch.setattr(runtime_releases, "_read_url", Mock(return_value=json.dumps(releases)))
    with pytest.raises(ValueError, match="invalid commit ID"):
        runtime_releases.latest_models_commit(catalog)
