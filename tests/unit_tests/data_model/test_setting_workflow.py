"""Tests for preparing and reusing parameter-setting inputs."""

from copy import deepcopy
from pathlib import Path

import pytest

from simtools.data_model import setting_workflow
from simtools.io import ascii_handler


@pytest.fixture
def setting_args(tmp_test_directory):
    return {
        "instrument": "LSTN-design",
        "parameter": "min_photons",
        "value": "120",
        "parameter_version": "2.0.1",
        "description": "Updated photon threshold",
        "output_path": tmp_test_directory / "settings",
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


def test_create_and_reuse_preserves_files(setting_args, setting_writer):
    config_file = setting_workflow.create_setting_workflow(setting_args)
    metadata_file = config_file.with_name("input.meta.yml")
    original = (config_file.read_bytes(), metadata_file.read_bytes())
    config = ascii_handler.collect_data_from_file(config_file)
    submission = config["applications"][0]["configuration"]
    assert submission["site"] == "North"
    assert submission["check_parameter_version"] is True
    assert submission["output_path"] == "output/__SETTING_WORKFLOW__/"
    metadata = ascii_handler.collect_data_from_file(metadata_file)["cta"]
    assert metadata["activity"]["id"] == config_file.parent.name
    assert metadata["product"]["description"] == setting_args["description"]
    assert metadata["product"]["data"]["model"]["version"] == "2.0.1"
    setting_args["value"] = "120.0"
    assert setting_workflow.create_setting_workflow(setting_args) == config_file
    assert (config_file.read_bytes(), metadata_file.read_bytes()) == original
    assert setting_writer.check_for_existing_parameter.call_count == 1


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


def test_explicit_site_for_shared_instrument(setting_args, setting_writer):
    setting_args.update(instrument="MSTx-NectarCam", site="South")
    config_file = setting_workflow.create_setting_workflow(setting_args)
    config = ascii_handler.collect_data_from_file(config_file)
    assert config["applications"][0]["configuration"]["site"] == "South"


def test_changed_metadata_is_preserved(setting_args, setting_writer):
    config_file = setting_workflow.create_setting_workflow(setting_args)
    metadata_file = config_file.with_name("input.meta.yml")
    metadata = ascii_handler.collect_data_from_file(metadata_file)
    metadata["cta"]["product"]["valid"] = {"start": "2026-01-01T00:00:00+00:00"}
    ascii_handler.write_data_to_file(metadata, metadata_file)
    previous = metadata_file.read_bytes()
    with pytest.raises(ValueError, match="Conflicting inputs"):
        setting_workflow.create_setting_workflow(setting_args)
    assert metadata_file.read_bytes() == previous


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
    assert not setting_args["output_path"].exists()


@pytest.mark.parametrize("parameter_type", ["file", "dict"])
def test_reject_file_or_structured_parameter(setting_args, setting_writer, parameter_type):
    setting_writer.get_parameter_type_for_schema.return_value = parameter_type
    with pytest.raises(ValueError, match="existing derivation workflow"):
        setting_workflow.create_setting_workflow(setting_args)
    assert not setting_args["output_path"].exists()


def test_repository_version_collision_leaves_no_workflow(setting_args, setting_writer):
    setting_writer.check_for_existing_parameter.side_effect = ValueError("already exists")
    with pytest.raises(ValueError, match="already exists"):
        setting_workflow.create_setting_workflow(setting_args)
    assert not setting_args["output_path"].exists()


def test_runtime_and_source_are_preserved(setting_args, setting_writer, tmp_test_directory):
    runtime_file = tmp_test_directory / "runtime.yml"
    runtime = {"runtime_environment": {"image": "example@sha256:123", "container_engine": "podman"}}
    ascii_handler.write_data_to_file(runtime, runtime_file)
    setting_args.update(workflow_runtime_file=runtime_file, source_url="https://example.org/data")
    config_file = setting_workflow.create_setting_workflow(setting_args)
    assert "runtime_environment" not in ascii_handler.collect_data_from_file(config_file)
    assert ascii_handler.collect_data_from_file(config_file.with_name("runtime.yml")) == runtime
    metadata = ascii_handler.collect_data_from_file(config_file.with_name("input.meta.yml"))
    assert metadata["cta"]["product"]["description"].endswith("Source: https://example.org/data")


def test_multiple_workflows_are_rejected(setting_args, setting_writer):
    config_file = setting_workflow.create_setting_workflow(setting_args)
    other = config_file.parent.parent / "other" / "config.yml"
    other.parent.mkdir()
    other.write_bytes(config_file.read_bytes())
    with pytest.raises(ValueError, match="Multiple workflows"):
        setting_workflow.create_setting_workflow(setting_args)


def test_run_and_rerun_use_separate_outputs(setting_args, setting_writer, mocker):
    config_file = setting_workflow.create_setting_workflow(setting_args)
    run = mocker.patch.object(setting_workflow.simtools_runner, "run_applications")
    output = setting_workflow.run_setting_workflow(config_file, {})
    assert output == setting_args["output_path"] / "output" / config_file.parent.relative_to(
        setting_args["output_path"] / "input"
    )
    assert run.call_args.kwargs["replacements"]["output/__SETTING_WORKFLOW__"] == str(output)
    assert run.call_args.args[0]["ignore_existing_parameter_version"] is False
    output.mkdir(parents=True)
    sentinel = output / "previous-result.json"
    sentinel.write_text("previous result", encoding="utf-8")
    rerun_output = setting_workflow.run_setting_workflow(config_file, {})
    assert rerun_output.parent == output / "reruns"
    assert run.call_args.args[0]["ignore_existing_parameter_version"] is True
    assert run.call_args.kwargs["replacements"]["output/__SETTING_WORKFLOW__"] == str(rerun_output)
    assert sentinel.read_text(encoding="utf-8") == "previous result"


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
