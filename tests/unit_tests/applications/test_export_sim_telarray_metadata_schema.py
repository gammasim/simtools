#!/usr/bin/python3

import argparse
import json
from types import SimpleNamespace

import yaml

from simtools.applications import export_sim_telarray_metadata_schema


def test_add_arguments_uses_suffix_based_output_format():
    parser = argparse.ArgumentParser()

    export_sim_telarray_metadata_schema._add_arguments(parser)

    args = parser.parse_args(["--output_file", "metadata.json"])
    assert args.output_file == "metadata.json"
    assert not hasattr(args, "output_format")


def test_main_writes_registry_using_output_file_suffix(tmp_test_directory, mocker):
    output_file = tmp_test_directory / "metadata.json"
    registry = {"name": "test", "meta_parameters": {"simtools_version": {"name": "test"}}}
    app_context = SimpleNamespace(
        args={
            "output_file": "metadata.json",
            "source_type": "generated",
            "schema_version": None,
        },
        io_handler=SimpleNamespace(get_output_file=lambda _: output_file),
        logger=SimpleNamespace(info=lambda _: None),
    )
    mocker.patch(
        "simtools.applications.export_sim_telarray_metadata_schema.build_application",
        return_value=app_context,
    )
    get_registry = mocker.patch(
        "simtools.applications.export_sim_telarray_metadata_schema."
        "simtel_validate_metadata.get_meta_parameter_registry",
        return_value=registry,
    )

    export_sim_telarray_metadata_schema.main()

    assert json.loads(output_file.read_text(encoding="utf-8")) == registry
    get_registry.assert_called_once_with(schema_version=None, source_type="generated")


def test_main_prints_yaml_to_stdout(capsys, mocker):
    registry = {"name": "test", "meta_parameters": {}}
    app_context = SimpleNamespace(
        args={"output_file": None, "source_type": "all", "schema_version": "latest"},
    )
    mocker.patch(
        "simtools.applications.export_sim_telarray_metadata_schema.build_application",
        return_value=app_context,
    )
    mocker.patch(
        "simtools.applications.export_sim_telarray_metadata_schema."
        "simtel_validate_metadata.get_meta_parameter_registry",
        return_value=registry,
    )

    export_sim_telarray_metadata_schema.main()

    assert yaml.safe_load(capsys.readouterr().out) == registry
