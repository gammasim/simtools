from pathlib import Path

from simtools.testing import resource_generation


def test_generate_test_resources_adds_integration_tests_placeholder(
    tmp_test_directory, monkeypatch
):
    called = {}

    monkeypatch.setattr(resource_generation, "validate_static_files", lambda _: None)
    monkeypatch.setattr(
        resource_generation,
        "get_resource_generation_directory",
        lambda *_: Path(tmp_test_directory) / "config_files",
    )
    monkeypatch.setattr(resource_generation, "download_files", lambda *_args, **_kwargs: [])

    def _fake_run_configured_applications(
        *, args_dict, config_dir, log_dir, run_time, replacements
    ):
        called["args_dict"] = args_dict
        called["config_dir"] = config_dir
        called["log_dir"] = log_dir
        called["run_time"] = run_time
        called["replacements"] = replacements

    monkeypatch.setattr(
        resource_generation, "run_configured_applications", _fake_run_configured_applications
    )

    resource_generation.generate_test_resources(
        {"test_directory": tmp_test_directory, "simtools_version": "v0.34.0"}
    )

    expected_integration_test_dir = resource_generation.get_integration_test_directory(
        tmp_test_directory, "v0.34.0"
    )

    assert called["replacements"] == {
        "__TEST_DIRECTORY__": str(Path(tmp_test_directory)),
        "__SIMTOOLS_VERSION__": "v0.34.0",
        "__INTEGRATION_TESTS_DIRECTORY__": str(expected_integration_test_dir),
    }
