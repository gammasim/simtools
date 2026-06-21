#!/usr/bin/env python3

import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from simtools.applications import generate_test_resource


def test_parse_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_test_resource.py",
            "--test_directory",
            "../simtools-tests",
            "--simtools_version",
            "v0.32.0",
            "--download_only",
        ],
    )

    args = generate_test_resource.parse_args()

    assert args.test_directory == Path("../simtools-tests")
    assert args.simtools_version == "v0.32.0"
    assert args.download_only is True
    assert not hasattr(args, "config_glob")


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

    monkeypatch.setattr(generate_test_resource.urllib.request, "urlretrieve", _fake_urlretrieve)

    resources_dir = Path(tmp_test_directory) / "resources"
    generate_test_resource.download_files(config_file=config_file, target_dir=resources_dir)

    assert (resources_dir / "folder" / "test.csv").exists()


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
        generate_test_resource.download_files(
            config_file=config_file, target_dir=tmp_test_directory
        )


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

    monkeypatch.setattr(generate_test_resource.urllib.request, "urlretrieve", _fail_urlretrieve)

    with pytest.raises(FileNotFoundError, match="Failed to download 'test file'"):
        generate_test_resource.download_files(
            config_file=config_file, target_dir=tmp_test_directory
        )


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

    monkeypatch.setattr("simtools.runners.simtools_runner.run_applications", _fake_run_applications)

    generate_test_resource.run_configured_applications(
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

    monkeypatch.setattr("simtools.runners.simtools_runner.run_applications", _fake_run_applications)

    generate_test_resource.run_configured_applications(
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
        generate_test_resource.get_resource_generation_directory(tmp_test_directory, "v0.32.0")
        == expected
    )


def test_get_resource_generation_directory_missing(tmp_test_directory):
    with pytest.raises(FileNotFoundError, match="Resource-generation directory"):
        generate_test_resource.get_resource_generation_directory(tmp_test_directory, "v0.32.0")


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

    monkeypatch.setattr(generate_test_resource.urllib.request, "urlretrieve", _fake_urlretrieve)

    generate_test_resource.download_files(
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
        generate_test_resource.download_files(
            config_file=config_file,
            target_dir=tmp_test_directory,
        )


def test_main_download_only_does_not_run_applications(tmp_test_directory, monkeypatch):
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
        generate_test_resource,
        "parse_args",
        lambda: SimpleNamespace(
            test_directory=Path(tmp_test_directory),
            simtools_version="v0.32.0",
            download_only=True,
            runtime_environment_file=None,
            ignore_runtime_environment=None,
            overwrite_collection_files=False,
        ),
    )
    monkeypatch.setattr(
        generate_test_resource,
        "run_configured_applications",
        lambda **kwargs: called.append(kwargs),
    )

    generate_test_resource.main()

    assert called == []


def test_main_prepares_shared_runtime_once(tmp_test_directory, monkeypatch):
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
    monkeypatch.setattr(
        generate_test_resource,
        "parse_args",
        lambda: SimpleNamespace(
            test_directory=Path(tmp_test_directory),
            simtools_version="v0.32.0",
            download_only=False,
            runtime_environment_file=runtime_file,
            ignore_runtime_environment=None,
            overwrite_collection_files=False,
        ),
    )

    def _prepare(path):
        prepare_mock.append(path)
        return {"image": "image"}, ["podman", "run", "image"]

    monkeypatch.setattr(generate_test_resource, "prepare_runtime_environment", _prepare)
    monkeypatch.setattr(
        generate_test_resource,
        "run_configured_applications",
        lambda **kwargs: run_calls.append(kwargs),
    )

    generate_test_resource.main()

    assert prepare_mock == [runtime_file]
    assert run_calls[0]["ignore_runtime_environment"] is False
    assert run_calls[0]["run_time"] == ["podman", "run", "image"]
