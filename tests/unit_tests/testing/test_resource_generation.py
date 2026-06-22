#!/usr/bin/env python3

import urllib.error
from pathlib import Path

import pytest

from simtools.testing import resource_generation


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

    monkeypatch.setattr(resource_generation.urllib.request, "urlretrieve", _fake_urlretrieve)

    resources_dir = Path(tmp_test_directory) / "resources"
    downloaded_files = resource_generation.download_files(
        config_file=config_file, target_dir=resources_dir
    )

    assert (resources_dir / "folder" / "test.csv").exists()
    assert downloaded_files == [resources_dir / "folder" / "test.csv"]


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
        resource_generation.download_files(config_file=config_file, target_dir=tmp_test_directory)


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

    monkeypatch.setattr(resource_generation.urllib.request, "urlretrieve", _fail_urlretrieve)

    with pytest.raises(FileNotFoundError, match="Failed to download 'test file'"):
        resource_generation.download_files(config_file=config_file, target_dir=tmp_test_directory)


def test_download_files_missing_configuration(tmp_test_directory):
    with pytest.raises(FileNotFoundError, match="Download configuration file does not exist"):
        resource_generation.download_files(
            Path(tmp_test_directory) / "missing.yml", tmp_test_directory
        )


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("- not-a-mapping\n", "must be a YAML mapping"),
        ("files: {}\n", "key 'files' must be a list"),
        ("gitlab_versions: []\nfiles: []\n", "key 'gitlab_versions' must be a dictionary"),
        ("files:\n- invalid-entry\n", "Download entry 0 must be a dictionary"),
    ],
)
def test_download_files_invalid_configuration(tmp_test_directory, content, match):
    config_file = Path(tmp_test_directory) / "download_files.yml"
    config_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        resource_generation.download_files(config_file, tmp_test_directory)


def test_run_configured_applications_runs_all_configs(tmp_test_directory, monkeypatch):
    config_root = Path(tmp_test_directory) / "config_files"
    model_parameters_dir = config_root / "model_parameters"
    application_config_dir = config_root / "application_config"
    model_parameters_dir.mkdir(parents=True)
    application_config_dir.mkdir(parents=True)

    model_config = model_parameters_dir / "array_element_position_ground.config.yml"
    app_config = application_config_dir / "simulate_prod.config.yml"
    model_config.write_text("steps: []\n", encoding="utf-8")
    app_config.write_text("steps: []\n", encoding="utf-8")

    called_configs = []

    def _fake_run_applications(config_dict, **kwargs):
        called_configs.append((config_dict, kwargs))

    monkeypatch.setattr(
        resource_generation.simtools_runner, "run_applications", _fake_run_applications
    )

    resource_generation.run_configured_applications(
        config_dir=config_root,
        log_dir=Path(tmp_test_directory) / "log_files",
        replacements={"__SIMTOOLS_VERSION__": "v0.32.0"},
    )

    assert [config[0]["config_file"] for config in called_configs] == [
        str(app_config),
        str(model_config),
    ]
    assert called_configs[0][0]["overwrite_collection_files"] is False
    assert called_configs[0][0]["log_file"] == str(
        Path(tmp_test_directory) / "log_files" / "simulate_prod.log"
    )
    assert called_configs[0][1]["replacements"] == {"__SIMTOOLS_VERSION__": "v0.32.0"}


def test_run_configured_applications_reuses_runtime(tmp_test_directory, monkeypatch):
    config_root = Path(tmp_test_directory) / "config_files"
    model_parameters_dir = config_root / "model_parameters"
    model_parameters_dir.mkdir(parents=True)

    model_config = model_parameters_dir / "mirror_list.config.yml"
    model_config.write_text("steps: []\n", encoding="utf-8")

    called_configs = []

    def _fake_run_applications(config_dict, **kwargs):
        called_configs.append((config_dict, kwargs))

    monkeypatch.setattr(
        resource_generation.simtools_runner, "run_applications", _fake_run_applications
    )

    resource_generation.run_configured_applications(
        config_dir=config_root,
        log_dir=Path(tmp_test_directory) / "log_files",
        ignore_runtime_environment=False,
        overwrite_collection_files=True,
        run_time=["podman", "run", "image"],
        runtime_environment={"image": "image"},
    )

    assert [config[0]["config_file"] for config in called_configs] == [str(model_config)]
    assert called_configs[0][0]["overwrite_collection_files"] is True
    assert called_configs[0][0]["runtime_environment"] == {"image": "image"}
    assert called_configs[0][1]["run_time"] == ["podman", "run", "image"]


def test_get_resource_generation_directory(tmp_test_directory):
    expected = (
        Path(tmp_test_directory)
        / "simtools-tests"
        / "v0.32.0"
        / "integration_tests"
        / "config_files"
    )
    expected.mkdir(parents=True)

    assert (
        resource_generation.get_resource_generation_directory(tmp_test_directory, "v0.32.0")
        == expected
    )


def test_get_resource_generation_directory_missing(tmp_test_directory):
    with pytest.raises(FileNotFoundError, match="Resource-generation directory"):
        resource_generation.get_resource_generation_directory(tmp_test_directory, "v0.32.0")


def test_download_files_replaces_version_placeholder(tmp_test_directory, monkeypatch):
    config_file = Path(tmp_test_directory) / "download_files.yml"
    config_file.write_text(
        "gitlab_versions:\n"
        "  simulation_model_parameter_setting: 0.1.0\n"
        "files:\n"
        "- url: https://example.org/__SIMULATION_MODEL_PARAMETER_SETTING_VERSION__/test.csv\n"
        "  description: test file\n"
        "  target_path: generated/test.csv\n",
        encoding="utf-8",
    )
    called_urls = []

    def _fake_urlretrieve(url, destination):
        called_urls.append(url)
        Path(destination).write_text("value\n", encoding="utf-8")

    monkeypatch.setattr(resource_generation.urllib.request, "urlretrieve", _fake_urlretrieve)

    resource_generation.download_files(
        config_file=config_file,
        target_dir=tmp_test_directory,
    )

    assert called_urls == ["https://example.org/0.1.0/test.csv"]


def test_download_files_requires_configured_version(tmp_test_directory):
    config_file = Path(tmp_test_directory) / "download_files.yml"
    config_file.write_text(
        "files:\n"
        "- url: https://example.org/__EXAMPLE_VERSION__/test.csv\n"
        "  description: test file\n"
        "  target_path: generated/test.csv\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No GitLab version configured"):
        resource_generation.download_files(
            config_file=config_file,
            target_dir=tmp_test_directory,
        )


def test_validate_static_files(tmp_test_directory):
    static_dir = Path(tmp_test_directory) / "static"
    static_dir.mkdir()
    (static_dir / "fixture.txt").write_text("abc", encoding="utf-8")
    manifest = static_dir / "static_manifest.yml"
    manifest.write_text(
        "files:\n"
        "- file_name: fixture.txt\n"
        "  sha256: ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n",
        encoding="utf-8",
    )

    resource_generation.validate_static_files(manifest)


def test_validate_static_files_reports_all_errors(tmp_test_directory):
    static_dir = Path(tmp_test_directory) / "static"
    static_dir.mkdir()
    (static_dir / "changed.txt").write_text("changed", encoding="utf-8")
    (static_dir / "unlisted.txt").write_text("unlisted", encoding="utf-8")
    manifest = static_dir / "static_manifest.yml"
    manifest.write_text(
        "files:\n"
        "- file_name: changed.txt\n"
        "  sha256: ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n"
        "- file_name: missing.txt\n"
        "  sha256: ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Checksum mismatch: changed.txt") as exc_info:
        resource_generation.validate_static_files(manifest)

    assert "Checksum mismatch: changed.txt" in str(exc_info.value)
    assert "Missing static file: missing.txt" in str(exc_info.value)
    assert "File not listed in manifest: unlisted.txt" in str(exc_info.value)


@pytest.mark.parametrize(
    ("content", "expected_exception", "match"),
    [
        (None, FileNotFoundError, "Static-file manifest does not exist"),
        ("files: {}\n", ValueError, "key 'files' must be a list"),
    ],
)
def test_validate_static_files_invalid_manifest(
    tmp_test_directory, content, expected_exception, match
):
    manifest = Path(tmp_test_directory) / "static_manifest.yml"
    if content is not None:
        manifest.write_text(content, encoding="utf-8")

    with pytest.raises(expected_exception, match=match):
        resource_generation.validate_static_files(manifest)


def test_validate_static_files_invalid_entries(tmp_test_directory):
    static_dir = Path(tmp_test_directory)
    (static_dir / "fixture.txt").write_text("abc", encoding="utf-8")
    manifest = static_dir / "static_manifest.yml"
    manifest.write_text(
        "files:\n"
        "- file_name: missing-checksum.txt\n"
        "- file_name: /absolute.txt\n"
        "  sha256: unused\n"
        "- file_name: fixture.txt\n"
        "  sha256: ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n"
        "- file_name: fixture.txt\n"
        "  sha256: ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Static-file validation failed") as exc_info:
        resource_generation.validate_static_files(manifest)

    assert "requires file_name and sha256" in str(exc_info.value)
    assert "Invalid manifest path: /absolute.txt" in str(exc_info.value)
    assert "Duplicate manifest entry: fixture.txt" in str(exc_info.value)


def test_prepare_runtime_environment(tmp_test_directory, monkeypatch):
    runtime_file = Path(tmp_test_directory) / "runtime.yml"
    runtime_file.write_text("runtime_environment:\n  image: test-image\n", encoding="utf-8")
    monkeypatch.setattr(
        resource_generation.simtools_runner,
        "read_runtime_environment",
        lambda config: ["runtime", config["image"]],
    )

    runtime_environment, run_time = resource_generation.prepare_runtime_environment(runtime_file)

    assert runtime_environment == {"image": "test-image"}
    assert run_time == ["runtime", "test-image"]


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("- invalid\n", "must be a YAML mapping"),
        ("other: value\n", "must contain a 'runtime_environment' block"),
    ],
)
def test_prepare_runtime_environment_invalid(tmp_test_directory, content, match):
    runtime_file = Path(tmp_test_directory) / "runtime.yml"
    runtime_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        resource_generation.prepare_runtime_environment(runtime_file)


def test_generate_test_resources_tests_static_files_only(tmp_test_directory, monkeypatch):
    integration_test_dir = (
        Path(tmp_test_directory) / "simtools-tests" / "v0.32.0" / "integration_tests"
    )
    manifest = integration_test_dir / "static" / "static_manifest.yml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("files: []\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        resource_generation,
        "validate_static_files",
        lambda path: calls.append(path),
    )
    monkeypatch.setattr(
        resource_generation,
        "download_files",
        lambda *_: pytest.fail("download_files must not be called"),
    )

    resource_generation.generate_test_resources(
        test_directory=tmp_test_directory,
        simtools_version="v0.32.0",
        test_static_files=True,
    )

    assert calls == [manifest]


def test_generate_test_resources_download_only_does_not_run_applications(
    tmp_test_directory, monkeypatch
):
    config_dir = (
        Path(tmp_test_directory)
        / "simtools-tests"
        / "v0.32.0"
        / "integration_tests"
        / "config_files"
    )
    config_dir.mkdir(parents=True)
    (config_dir / "download_files.yml").write_text("files: []\n", encoding="utf-8")
    called = []
    monkeypatch.setattr(
        resource_generation,
        "run_configured_applications",
        lambda **kwargs: called.append(kwargs),
    )

    resource_generation.generate_test_resources(
        test_directory=tmp_test_directory,
        simtools_version="v0.32.0",
        download_only=True,
    )

    assert called == []


def test_generate_test_resources_prepares_shared_runtime_once(tmp_test_directory, monkeypatch):
    config_dir = (
        Path(tmp_test_directory)
        / "simtools-tests"
        / "v0.32.0"
        / "integration_tests"
        / "config_files"
    )
    config_dir.mkdir(parents=True)
    (config_dir / "download_files.yml").write_text("files: []\n", encoding="utf-8")
    runtime_file = Path(tmp_test_directory) / "runtime.yml"
    prepare_mock = []
    run_calls = []

    def _prepare(path):
        prepare_mock.append(path)
        return {"image": "image"}, ["podman", "run", "image"]

    monkeypatch.setattr(resource_generation, "prepare_runtime_environment", _prepare)
    monkeypatch.setattr(
        resource_generation,
        "run_configured_applications",
        lambda **kwargs: run_calls.append(kwargs),
    )

    resource_generation.generate_test_resources(
        test_directory=tmp_test_directory,
        simtools_version="v0.32.0",
        runtime_environment_file=runtime_file,
    )

    assert prepare_mock == [runtime_file]
    assert run_calls[0]["ignore_runtime_environment"] is False
    assert run_calls[0]["run_time"] == ["podman", "run", "image"]


def test_generate_test_resources_ignores_supplied_runtime(tmp_test_directory, monkeypatch):
    config_dir = (
        Path(tmp_test_directory)
        / "simtools-tests"
        / "v0.32.0"
        / "integration_tests"
        / "config_files"
    )
    config_dir.mkdir(parents=True)
    (config_dir / "download_files.yml").write_text("files: []\n", encoding="utf-8")
    run_calls = []
    monkeypatch.setattr(
        resource_generation,
        "prepare_runtime_environment",
        lambda _: pytest.fail("runtime must not be prepared"),
    )
    monkeypatch.setattr(
        resource_generation,
        "run_configured_applications",
        lambda **kwargs: run_calls.append(kwargs),
    )

    resource_generation.generate_test_resources(
        test_directory=tmp_test_directory,
        simtools_version="v0.32.0",
        runtime_environment_file=Path(tmp_test_directory) / "runtime.yml",
        ignore_runtime_environment=True,
    )

    assert run_calls[0]["ignore_runtime_environment"] is True
    assert run_calls[0]["run_time"] is None


def test_remove_empty_download_directories_edge_cases(tmp_test_directory):
    target_dir = Path(tmp_test_directory) / "integration_tests"
    removable_dir = target_dir / "temporary"
    preserved_dir = target_dir / "generated"
    nonempty_dir = target_dir / "nonempty"
    removable_dir.mkdir(parents=True)
    preserved_dir.mkdir()
    nonempty_dir.mkdir()
    (nonempty_dir / "kept.txt").write_text("content", encoding="utf-8")

    resource_generation._remove_empty_download_directories(
        [
            Path(tmp_test_directory) / "outside" / "file.txt",
            target_dir / "root-file.txt",
            removable_dir / "removed.txt",
            preserved_dir / "preserved.txt",
            nonempty_dir / "removed.txt",
        ],
        target_dir,
    )

    assert not removable_dir.exists()
    assert preserved_dir.is_dir()
    assert nonempty_dir.is_dir()


def test_generate_test_resources_removes_empty_download_directory(tmp_test_directory, monkeypatch):
    integration_test_dir = (
        Path(tmp_test_directory) / "simtools-tests" / "v0.32.0" / "integration_tests"
    )
    config_dir = integration_test_dir / "config_files"
    config_dir.mkdir(parents=True)
    (config_dir / "download_files.yml").write_text("files: []\n", encoding="utf-8")
    downloaded_file = integration_test_dir / "folder" / "input.dat"

    def _download_files(*_):
        downloaded_file.parent.mkdir()
        downloaded_file.write_text("input", encoding="utf-8")
        return [downloaded_file]

    def _run_configured_applications(**_):
        downloaded_file.unlink()

    monkeypatch.setattr(resource_generation, "download_files", _download_files)
    monkeypatch.setattr(
        resource_generation, "run_configured_applications", _run_configured_applications
    )

    resource_generation.generate_test_resources(
        test_directory=tmp_test_directory,
        simtools_version="v0.32.0",
    )

    assert not (integration_test_dir / "folder").exists()
