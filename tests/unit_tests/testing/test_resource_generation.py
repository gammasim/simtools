#!/usr/bin/env python3

import urllib.error
from pathlib import Path

import pytest

from simtools.testing import resource_generation

DOWNLOAD_CONFIG = """
base_urls:
  test_base:
    url: https://example.org/
    version: v1.0.0
files:
  - base_url_key: test_base
    path: test.csv
    description: test file
    target_path: folder/test.csv
"""


def _write_config(directory, content, name="download_files.yml"):
    config_file = Path(directory) / name
    config_file.write_text(content.strip() + "\n", encoding="utf-8")
    return config_file


def _fake_run(calls):
    def _run(config_dict, **kwargs):
        calls.append((config_dict, kwargs))

    return _run


def _integration_config_dir(root, version="v0.32.0"):
    config_dir = Path(root) / "simtools-tests" / version / "integration_tests" / "config_files"
    config_dir.mkdir(parents=True)
    (config_dir / "download_files.yml").write_text("files: []\n", encoding="utf-8")
    return config_dir


def test_download_files(tmp_test_directory, monkeypatch):
    config_file = _write_config(tmp_test_directory, DOWNLOAD_CONFIG)

    def _fake_urlretrieve(url, destination):
        assert url == "https://example.org/v1.0.0/test.csv"
        Path(destination).write_text("value\n", encoding="utf-8")
        return destination, None

    monkeypatch.setattr(resource_generation.urllib.request, "urlretrieve", _fake_urlretrieve)
    resources_dir = Path(tmp_test_directory) / "resources"
    downloaded_files = resource_generation.download_files(
        config_file=config_file, target_dir=resources_dir
    )
    assert (resources_dir / "folder" / "test.csv").exists()
    assert downloaded_files == [resources_dir / "folder" / "test.csv"]


def test_download_files_download_failure(tmp_test_directory, monkeypatch):
    config_file = _write_config(tmp_test_directory, DOWNLOAD_CONFIG)

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
        ("base_urls: []\nfiles: []\n", "key 'base_urls' must be a dictionary"),
        ("files:\n- invalid-entry\n", "Download entry 0 must be a dictionary"),
        (
            DOWNLOAD_CONFIG.replace("    target_path: folder/test.csv\n", ""),
            "missing required keys: target_path",
        ),
        (
            DOWNLOAD_CONFIG.replace("base_url_key: test_base", "base_url_key: unknown_base"),
            "Base URL key 'unknown_base' not found",
        ),
    ],
)
def test_download_files_invalid_configuration(tmp_test_directory, content, match):
    config_file = _write_config(tmp_test_directory, content)
    with pytest.raises(ValueError, match=match):
        resource_generation.download_files(config_file, tmp_test_directory)


def test_run_configured_applications_runs_all_configs(tmp_test_directory, monkeypatch):
    config_root = Path(tmp_test_directory) / "config_files"
    model_config = config_root / "model_parameters" / "array_element_position_ground.config.yml"
    app_config = config_root / "application_config" / "simulate_prod.config.yml"
    for path in (model_config, app_config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("steps: []\n", encoding="utf-8")
    called_configs = []
    monkeypatch.setattr(
        resource_generation.simtools_runner, "run_applications", _fake_run(called_configs)
    )
    replacements = {
        "__SIMTOOLS_VERSION__": "v0.32.0",
        "__INTEGRATION_TESTS_DIRECTORY__": str(Path(tmp_test_directory) / "integration_tests"),
    }
    resource_generation.run_configured_applications(
        args_dict={},
        config_dir=config_root,
        log_dir=Path(tmp_test_directory) / "log_files",
        run_time=None,
        replacements=replacements,
    )

    assert [config[0]["config_file"] for config in called_configs] == [
        str(app_config),
        str(model_config),
    ]
    assert called_configs[0][0]["overwrite_collection_files"] is False
    assert called_configs[0][0]["log_file"] == str(
        Path(tmp_test_directory) / "log_files" / "simulate_prod.log"
    )
    assert called_configs[0][1]["replacements"] == replacements


def test_run_configured_applications_unique_tmp_and_runtime(tmp_test_directory, monkeypatch):
    config_root = Path(tmp_test_directory) / "config_files"
    (config_root / "model_parameters").mkdir(parents=True)
    (config_root / "model_parameters" / "mirror_list.config.yml").write_text(
        "steps: []\n", encoding="utf-8"
    )
    called_configs = []
    monkeypatch.setattr(
        resource_generation.simtools_runner, "run_applications", _fake_run(called_configs)
    )
    resource_generation.run_configured_applications(
        args_dict={
            "ignore_runtime_environment": False,
            "overwrite_collection_files": True,
            "runtime_environment": {"image": "image"},
        },
        config_dir=config_root,
        log_dir=Path(tmp_test_directory) / "log_files",
        run_time=["podman", "run", "image"],
        replacements={"__INTEGRATION_TESTS_DIRECTORY__": str(tmp_test_directory)},
        unique_tmp_directories=True,
    )
    config, kwargs = called_configs[0]
    assert config["overwrite_collection_files"] is True
    assert config["runtime_environment"] == {"image": "image"}
    assert kwargs["run_time"] == ["podman", "run", "image"]


def test_run_configured_applications_uses_unique_tmp_directory(tmp_test_directory, monkeypatch):
    config_root = Path(tmp_test_directory) / "config_files"
    config_root.mkdir(parents=True)
    (config_root / "simulate_prod.config.yml").write_text("steps: []\n", encoding="utf-8")
    called = []
    monkeypatch.setattr(
        resource_generation.simtools_runner,
        "run_applications",
        lambda config, **kwargs: called.append(kwargs),
    )
    integration_dir = Path(tmp_test_directory) / "integration_tests"
    resource_generation.run_configured_applications(
        args_dict={},
        config_dir=config_root,
        log_dir=integration_dir / "log_files",
        run_time=None,
        replacements={
            "__INTEGRATION_TESTS_DIRECTORY__": str(integration_dir),
            "__INTEGRATION_TESTS_DIRECTORY__/tmp/": "stale/tmp/",
        },
        unique_tmp_directories=True,
    )
    called_config = next(iter(called), None)
    assert called_config is not None, "The workflow application was not called"
    replacement = Path(called_config["replacements"]["__INTEGRATION_TESTS_DIRECTORY__/tmp/"])
    assert replacement.parent == integration_dir / "tmp"
    assert replacement.name.startswith("simulate_prod.config-")
    assert replacement.is_relative_to(Path(tmp_test_directory))
    assert called_config["replacements"]["__INTEGRATION_TESTS_DIRECTORY__"] == str(integration_dir)


def test_run_configured_applications_selects_one_config_by_path(tmp_test_directory, monkeypatch):
    config_root = Path(tmp_test_directory) / "config_files"
    selected_config = config_root / "application_config" / "simulate_prod.config.yml"
    other_config = config_root / "model_parameters" / "mirror_list.config.yml"
    for path in (selected_config, other_config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("steps: []\n", encoding="utf-8")
    called_configs = []
    monkeypatch.setattr(
        resource_generation.simtools_runner, "run_applications", _fake_run(called_configs)
    )
    resource_generation.run_configured_applications(
        args_dict={"config_file": Path("application_config") / "simulate_prod.config.yml"},
        config_dir=config_root,
        log_dir=Path(tmp_test_directory) / "log_files",
        run_time=None,
        replacements=None,
    )
    assert [config[0]["config_file"] for config in called_configs] == [
        str(selected_config.resolve())
    ]

    with pytest.raises(FileNotFoundError, match="Selected workflow config does not exist"):
        resource_generation.run_configured_applications(
            args_dict={"config_file": Path("missing.config.yml")},
            config_dir=config_root,
            log_dir=Path(tmp_test_directory) / "log_files",
            run_time=None,
            replacements=None,
        )


def test_get_resource_generation_directory(tmp_test_directory):
    with pytest.raises(FileNotFoundError, match="Resource-generation directory"):
        resource_generation.get_resource_generation_directory(tmp_test_directory, "v0.32.0")

    expected = _integration_config_dir(tmp_test_directory)
    assert (
        resource_generation.get_resource_generation_directory(tmp_test_directory, "v0.32.0")
        == expected
    )


def test_initialize_test_resources(tmp_test_directory):
    template_dir = Path(tmp_test_directory) / "simtools-tests" / "v0.36.0" / "integration_tests"
    (template_dir / "config_files").mkdir(parents=True)
    (template_dir / "static").mkdir()
    (template_dir / "generated").mkdir()
    (template_dir / "config_files" / "workflow.config.yml").write_text(
        "steps: []\n", encoding="utf-8"
    )
    (template_dir / "static" / "static_manifest.yml").write_text("files: []\n", encoding="utf-8")
    (template_dir / "run_time.yml").write_text("runtime_environment: {}\n", encoding="utf-8")

    target_dir = resource_generation.initialize_test_resources(
        tmp_test_directory, "v0.37.0", "v0.36.0"
    )

    assert target_dir == (
        Path(tmp_test_directory) / "simtools-tests" / "v0.37.0" / "integration_tests"
    )
    assert (target_dir / "config_files" / "workflow.config.yml").is_file()
    assert (target_dir / "static" / "static_manifest.yml").is_file()
    assert (target_dir / "run_time.yml").is_file()
    assert not (target_dir / "generated").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        resource_generation.initialize_test_resources(tmp_test_directory, "v0.37.0", "v0.36.0")


SHA_FIXTURE = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_validate_static_files(tmp_test_directory):
    static_dir = Path(tmp_test_directory) / "static"
    static_dir.mkdir()
    (static_dir / "fixture.txt").write_text("abc", encoding="utf-8")
    manifest = static_dir / "static_manifest.yml"
    manifest.write_text(
        f"files:\n- file_name: fixture.txt\n  sha256: {SHA_FIXTURE}\n", encoding="utf-8"
    )
    assert resource_generation.validate_static_files(manifest) is None


def test_validate_static_files_reports_all_errors(tmp_test_directory):
    static_dir = Path(tmp_test_directory) / "static"
    static_dir.mkdir()
    (static_dir / "changed.txt").write_text("changed", encoding="utf-8")
    (static_dir / "unlisted.txt").write_text("unlisted", encoding="utf-8")
    manifest = static_dir / "static_manifest.yml"
    manifest.write_text(
        f"files:\n- file_name: changed.txt\n  sha256: {SHA_FIXTURE}\n"
        f"- file_name: missing.txt\n  sha256: {SHA_FIXTURE}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"Checksum mismatch: changed.txt") as exc_info:
        resource_generation.validate_static_files(manifest)
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
        f"- file_name: fixture.txt\n  sha256: {SHA_FIXTURE}\n"
        f"- file_name: fixture.txt\n  sha256: {SHA_FIXTURE}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Static-file validation failed") as exc_info:
        resource_generation.validate_static_files(manifest)
    assert "requires file_name and sha256" in str(exc_info.value)
    assert "Invalid manifest path: /absolute.txt" in str(exc_info.value)
    assert "Duplicate manifest entry: fixture.txt" in str(exc_info.value)


def test_generate_test_resources_static_files_only(tmp_test_directory, monkeypatch):
    manifest = (
        Path(tmp_test_directory)
        / "simtools-tests"
        / "v0.32.0"
        / "integration_tests"
        / "static"
        / "static_manifest.yml"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text("files: []\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(resource_generation, "validate_static_files", calls.append)
    monkeypatch.setattr(
        resource_generation,
        "download_files",
        lambda *_: pytest.fail("download_files must not be called"),
    )
    resource_generation.generate_test_resources(
        args_dict={
            "test_directory": tmp_test_directory,
            "simtools_version": "v0.32.0",
            "test_static_files": True,
        }
    )
    assert calls == [manifest]


def test_generate_test_resources_download_only_does_not_run_applications(
    tmp_test_directory, monkeypatch
):
    _integration_config_dir(tmp_test_directory)
    called = []
    monkeypatch.setattr(
        resource_generation,
        "run_configured_applications",
        lambda **kwargs: called.append(kwargs),
    )
    resource_generation.generate_test_resources(
        args_dict={
            "test_directory": tmp_test_directory,
            "simtools_version": "v0.32.0",
            "download_only": True,
        }
    )
    assert called == []


def test_generate_test_resources_prepares_shared_runtime_once(tmp_test_directory, monkeypatch):
    _integration_config_dir(tmp_test_directory)
    prepare_calls = []
    run_calls = []

    def _prepare(path):
        prepare_calls.append(path)
        return {"image": "image"}, ["podman", "run", "image"]

    monkeypatch.setattr(
        resource_generation.simtools_runner, "prepare_runtime_environment", _prepare
    )
    monkeypatch.setattr(
        resource_generation,
        "run_configured_applications",
        lambda **kwargs: run_calls.append(kwargs),
    )
    resource_generation.generate_test_resources(
        args_dict={"test_directory": tmp_test_directory, "simtools_version": "v0.32.0"},
        run_time=["podman", "run", "image"],
    )

    assert len(run_calls) == 1
    assert run_calls[0]["run_time"] == ["podman", "run", "image"]
    assert run_calls[0]["unique_tmp_directories"] is True


def test_generate_test_resources_forwards_selected_config_file(tmp_test_directory, monkeypatch):
    _integration_config_dir(tmp_test_directory)
    run_calls = []
    selected_config = Path("application_config") / "simulate_prod.config.yml"
    monkeypatch.setattr(
        resource_generation,
        "run_configured_applications",
        lambda **kwargs: run_calls.append(kwargs),
    )
    resource_generation.generate_test_resources(
        args_dict={
            "test_directory": tmp_test_directory,
            "simtools_version": "v0.32.0",
            "config_file": selected_config,
        }
    )
    assert len(run_calls) == 1
    assert run_calls[0]["args_dict"]["config_file"] == selected_config


def test_remove_empty_download_directories(tmp_test_directory):
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
