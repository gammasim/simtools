"""Tests for preparing and reusing parameter-setting inputs."""

from copy import deepcopy
from pathlib import Path

import pytest

from simtools.io import ascii_handler
from simtools.runners import setting_workflow


@pytest.fixture
def setting_args(tmp_test_directory):
    output_path = Path(tmp_test_directory) / "settings"
    output_path.mkdir(parents=True)
    ascii_handler.write_data_to_file(
        {"runtime_environment": {"image": "default@sha256:abc"}},
        output_path / "runtime.default.yml",
    )
    return {
        "instrument": "LSTN-design",
        "parameter": "min_photons",
        "value": "120",
        "parameter_version": "2.0.1",
        "description": "Updated photon threshold",
        "output_path": output_path,
    }


@pytest.fixture
def setting_writer(mocker):
    writer = mocker.patch.object(setting_workflow, "ModelDataWriter").return_value
    writer.get_parameter_type_for_schema.return_value = "float64"
    writer.get_validated_parameter_dict.side_effect = lambda **args: {
        "value": float(args["value"]),
        "unit": None,
        "model_parameter_schema_version": "0.1.0",
    }
    mocker.patch.object(setting_workflow.schema, "validate_dict_using_schema")
    collector = mocker.patch.object(setting_workflow, "MetadataCollector").return_value
    collector.get_top_level_metadata.side_effect = lambda: deepcopy(
        {
            "cta": {
                "activity": {"id": "generated"},
                "contact": {"name": "Observer"},
                "instrument": {"ID": "LSTN-design"},
                "product": {"data": {"model": {}}, "id": "product-id"},
            }
        }
    )
    return writer


@pytest.mark.parametrize(
    "change",
    [
        {"value": "121"},
        {"description": "Another reason"},
        {"source_url": "https://example.org/measurement"},
    ],
)
def test_conflicting_input_is_preserved(setting_args, setting_writer, change):
    config_file = setting_workflow.create_setting_workflow(setting_args)
    original = config_file.read_bytes()
    setting_args.update(change)
    with pytest.raises(ValueError, match=r"Conflicting inputs.*new parameter version"):
        setting_workflow.create_setting_workflow(setting_args)
    assert config_file.read_bytes() == original
    assert len(list(config_file.parent.parent.iterdir())) == 1


def test_new_version_gets_new_workflow(setting_args, setting_writer):
    previous = setting_workflow.create_setting_workflow(setting_args)
    setting_args.update(parameter_version="2.0.2", value="121")
    new = setting_workflow.create_setting_workflow(setting_args)
    assert new != previous
    assert previous.is_file()


def test_changed_runtime_cannot_reuse_existing_workflow(setting_args, setting_writer):
    config_file = setting_workflow.create_setting_workflow(setting_args)
    runtime_file = config_file.with_name("runtime.yml")
    original = runtime_file.read_bytes()
    ascii_handler.write_data_to_file(
        {"runtime_environment": {"image": "different-image"}},
        setting_args["output_path"] / "runtime.default.yml",
    )
    with pytest.raises(ValueError, match=r"Conflicting inputs.*new parameter version"):
        setting_workflow.create_setting_workflow(setting_args)
    assert runtime_file.read_bytes() == original


def test_missing_existing_runtime_cannot_reuse_workflow(setting_args, setting_writer):
    config_file = setting_workflow.create_setting_workflow(setting_args)
    config_file.with_name("runtime.yml").unlink()
    with pytest.raises(ValueError, match=r"Conflicting inputs.*new parameter version"):
        setting_workflow.create_setting_workflow(setting_args)


def test_explicit_site_for_shared_instrument(setting_args, setting_writer):
    setting_args.update(instrument="MSTx-NectarCam", site="South")
    config_file = setting_workflow.create_setting_workflow(setting_args)
    config = ascii_handler.collect_data_from_file(config_file)
    assert config["applications"][0]["configuration"]["site"] == "South"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"parameter_version": "v2.0.1"}, "semantic version"),
        ({"description": " "}, "description"),
        ({"instrument": "invalid"}, "Invalid"),
        ({"site": "South"}, "does not match"),
        ({"instrument": "MSTx-NectarCam"}, "Specify --site"),
        ({"parameter": "../min_photons"}, "without path separators"),
    ],
)
def test_invalid_inputs_leave_no_workflow(setting_args, setting_writer, change, message):
    setting_args.update(change)
    with pytest.raises(ValueError, match=message):
        setting_workflow.create_setting_workflow(setting_args)
    assert not (setting_args["output_path"] / "input").exists()


@pytest.mark.parametrize("parameter_type", ["file", "dict"])
def test_reject_file_or_structured_parameter(setting_args, setting_writer, parameter_type):
    setting_writer.get_parameter_type_for_schema.return_value = parameter_type
    with pytest.raises(ValueError, match="existing derivation workflow"):
        setting_workflow.create_setting_workflow(setting_args)
    assert not (setting_args["output_path"] / "input").exists()


def test_run_uses_workflow_runtime_file(setting_args, setting_writer, mocker):
    runtime_file = Path(str(setting_args["output_path"])).parent / "runtime.yml"
    ascii_handler.write_data_to_file(
        {"runtime_environment": {"image": "example@sha256:123"}}, runtime_file
    )
    setting_args["workflow_runtime_file"] = runtime_file
    config_file = setting_workflow.create_setting_workflow(setting_args)
    prepare = mocker.patch.object(
        setting_workflow.simtools_runner,
        "prepare_runtime_environment",
        return_value=({"image": "example@sha256:123"}, ["runtime"]),
    )
    run = mocker.patch.object(setting_workflow.simtools_runner, "run_applications")

    setting_workflow.run_setting_workflow(config_file, {})

    prepare.assert_called_once_with(config_file.with_name("runtime.yml"))
    assert run.call_args.args[0]["runtime_environment"] == {"image": "example@sha256:123"}
    assert run.call_args.kwargs["run_time"] == ["runtime"]


def test_default_runtime_file_is_copied_to_workflow(setting_args, setting_writer):
    config_file = setting_workflow.create_setting_workflow(setting_args)
    runtime = ascii_handler.collect_data_from_file(config_file.with_name("runtime.yml"))
    assert runtime == {"runtime_environment": {"image": "default@sha256:abc"}}


def test_default_runtime_resolves_unset_releases(setting_args, setting_writer, mocker):
    default_file = setting_args["output_path"] / "runtime.default.yml"
    ascii_handler.write_data_to_file(
        {"runtime_environment": {"image": None, "simulation_models": {"git_revision": None}}},
        default_file,
    )
    mocker.patch.object(
        setting_workflow.runtime_releases, "latest_simtools_image", return_value="image@sha256:123"
    )
    mocker.patch.object(
        setting_workflow.runtime_releases, "latest_models_commit", return_value="abc123"
    )
    config_file = setting_workflow.create_setting_workflow(setting_args)
    runtime = ascii_handler.collect_data_from_file(config_file.with_name("runtime.yml"))
    assert runtime == {
        "runtime_environment": {
            "image": "image@sha256:123",
            "simulation_models": {"git_revision": "abc123"},
        }
    }
    assert (
        ascii_handler.collect_data_from_file(default_file)["runtime_environment"]["image"] is None
    )


def test_empty_default_resolves_both_releases(mocker):
    image = mocker.patch.object(
        setting_workflow.runtime_releases, "latest_simtools_image", return_value="image@sha256:123"
    )
    models = mocker.patch.object(
        setting_workflow.runtime_releases, "latest_models_commit", return_value="abc123"
    )
    runtime = {"runtime_environment": {}}
    setting_workflow._resolve_default_releases(runtime)
    assert runtime["runtime_environment"]["simulation_models"]["git_revision"] == "abc123"
    image.assert_called_once()
    models.assert_called_once()


def test_release_lookup_failure_explains_how_to_proceed(setting_args, setting_writer, mocker):
    ascii_handler.write_data_to_file(
        {"runtime_environment": {"image": None}},
        setting_args["output_path"] / "runtime.default.yml",
    )
    mocker.patch.object(
        setting_workflow.runtime_releases, "latest_simtools_image", side_effect=OSError("offline")
    )
    with pytest.raises(ValueError, match="Check network access or set"):
        setting_workflow.create_setting_workflow(setting_args)


def test_explicit_runtime_file_overrides_default(setting_args, setting_writer, tmp_test_directory):
    explicit = Path(tmp_test_directory) / "explicit-runtime.yml"
    ascii_handler.write_data_to_file(
        {"runtime_environment": {"image": "explicit@sha256:def"}}, explicit
    )
    setting_args["workflow_runtime_file"] = explicit
    config_file = setting_workflow.create_setting_workflow(setting_args)
    runtime = ascii_handler.collect_data_from_file(config_file.with_name("runtime.yml"))
    assert runtime == {"runtime_environment": {"image": "explicit@sha256:def"}}


def test_missing_default_runtime_file_raises(setting_args, setting_writer):
    (setting_args["output_path"] / "runtime.default.yml").unlink()
    with pytest.raises(FileNotFoundError, match=r"runtime\.default\.yml.*--workflow_runtime_file"):
        setting_workflow.create_setting_workflow(setting_args)
    assert not (setting_args["output_path"] / "input").exists()


def test_inline_runtime_and_runtime_file_conflict(setting_args, setting_writer, mocker):
    config_file = setting_workflow.create_setting_workflow(setting_args)
    config = ascii_handler.collect_data_from_file(config_file)
    config["runtime_environment"] = {"image": "inline@sha256:123"}
    ascii_handler.write_data_to_file(config, config_file)
    run = mocker.patch.object(setting_workflow.simtools_runner, "run_applications")
    with pytest.raises(ValueError, match=r"runtime_environment block.*runtime\.yml"):
        setting_workflow.run_setting_workflow(config_file, {})
    run.assert_not_called()
