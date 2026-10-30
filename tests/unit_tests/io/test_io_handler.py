#!/usr/bin/python3

from pathlib import Path

from simtools.io import io_handler as io_handler_module


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


def test_resolve_test_resource_paths_requires_explicit_resource_root(
    tmp_test_directory, monkeypatch
):
    resources_path = tmp_test_directory / "resources"
    monkeypatch.setenv("SIMTOOLS_TESTS_PATH", str(tmp_test_directory / "ignored"))

    resolved = io_handler_module.resolve_test_resource_paths(
        "${generated:input.ecsv}", resources_path
    )

    assert resolved == str(resources_path / "generated/input.ecsv")
