#!/usr/bin/python3

import copy
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from simtools.io import io_handler as io_handler_module

logger = logging.getLogger()

test_file = "test-file.txt"


def test_get_output_directory(args_dict, io_handler):
    # default adding label
    assert io_handler.get_output_directory() == Path(f"{args_dict['output_path']}/output/")

    # label and subdirectory
    assert io_handler.get_output_directory(sub_dir="model") == Path(
        f"{args_dict['output_path']}/output/model/"
    )

    # path ends with '-output' - no additional 'output' is added
    io_handler_copy = copy.deepcopy(io_handler)
    io_handler_copy.output_path["default"] = Path(f"{args_dict['output_path']}/unittest-output")
    assert io_handler_copy.get_output_directory(sub_dir="model") == Path(
        f"{args_dict['output_path']}/unittest-output/model"
    )

    # FileNotFoundError
    with patch.object(Path, "mkdir", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError, match=r"^Error creating directory"):
            io_handler.get_output_directory(sub_dir="model")

    # non existing path
    with pytest.raises(KeyError, match=r"Output path label 'nonexistent' not found"):
        io_handler.get_output_directory(output_path_label="nonexistent")


def test_get_test_data_file(io_handler, test_resources_path):
    assert (
        io_handler.get_test_data_file(file_name=test_file)
        == (test_resources_path / test_file).resolve()
    )


def test_resolve_test_resource_path_macros_nested_structures(tmp_test_directory):
    resources_path = Path(tmp_test_directory) / "versioned-resources"
    value = {
        "config": {
            "input": "${static:data/file.ecsv}",
            "groups": [
                "prefix/${generated:run/output.fits}",
                {
                    "items": [
                        "${static:layout/array.ecsv}",
                        "${generated:plots/summary.png}",
                        "${downloaded:metadata/asum_threshold.meta.yml}",
                    ],
                    "keep_number": 7,
                    "keep_bool": True,
                },
            ],
        },
        "plain": "no_macro",
    }

    resolved = io_handler_module.resolve_test_resource_paths(value, resources_path)

    assert resolved["config"]["input"] == str(resources_path / "static/data/file.ecsv")
    assert resolved["config"]["groups"][0] == f"prefix/{resources_path}/generated/run/output.fits"
    assert resolved["config"]["groups"][1]["items"][0] == str(
        resources_path / "static/layout/array.ecsv"
    )
    assert resolved["config"]["groups"][1]["items"][1] == str(
        resources_path / "generated/plots/summary.png"
    )
    assert resolved["config"]["groups"][1]["items"][2] == str(
        resources_path / "downloaded/metadata/asum_threshold.meta.yml"
    )
    assert resolved["config"]["groups"][1]["keep_number"] == 7
    assert resolved["config"]["groups"][1]["keep_bool"] is True
    assert resolved["plain"] == "no_macro"


def test_resolve_test_resource_paths_uses_versioned_environment_resources(
    tmp_test_directory, monkeypatch
):
    tests_path = tmp_test_directory / "simtools-tests"
    monkeypatch.delenv("SIMTOOLS_TEST_RESOURCES", raising=False)
    monkeypatch.setenv("SIMTOOLS_TESTS_PATH", str(tests_path))
    monkeypatch.setenv("SIMTOOLS_TESTS_TAG", "v0.37.0")

    resolved = io_handler_module.resolve_test_resource_paths("${downloaded:corsika_limits.ecsv}")

    expected_root = tests_path / "v0.37.0" / "integration_tests"
    assert resolved == str(expected_root / "downloaded/corsika_limits.ecsv")


def test_get_model_configuration_directory(args_dict, io_handler):
    model_version = "1.0.0"
    label = "test-io-handler"

    # Test directory creation
    expected_path = Path(f"{args_dict['output_path']}/output/model/{label}/{model_version}")
    assert (
        io_handler.get_model_configuration_directory(sub_dir=label, model_version=model_version)
        == expected_path
    )

    # Test FileNotFoundError
    with patch.object(Path, "mkdir", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError, match=r"^Error creating directory"):
            io_handler.get_model_configuration_directory(sub_dir=label, model_version=model_version)
