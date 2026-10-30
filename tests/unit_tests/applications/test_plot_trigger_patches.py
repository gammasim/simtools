"""Tests for configuration-only trigger-patch plotting application."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from astropy.table import Table

from simtools.applications import plot_trigger_patches
from simtools.visualization.plot_trigger_patches import format_trigger_patch_summary

GROUP_ID = "group_id"
PIXEL_ID = "pixel_id"
PIXEL_IDS = "pix_id"
SITE_NAME = "North"
TELESCOPE_NAME = "LSTN-test"
MODEL_VERSION_VALUE = "1.0.0"
SITE_OPTION = "--site"
TELESCOPE_OPTION = "--telescope"
MODEL_VERSION_OPTION = "--model_version"
SUMMARY_FILE_KEY = "summary_file"
PIXELS_FILE_KEY = "pixels_file"
GROUPS_FILE_KEY = "groups_file"
ECSV_FORMAT = "ascii.ecsv"


def test_application_does_not_resolve_simulation_executables():
    assert "corsika" not in plot_trigger_patches.APPLICATION.needs
    assert "sim_telarray" not in plot_trigger_patches.APPLICATION.needs


def test_cli_requires_model_inputs_and_has_no_simulation_stimulus_options(capsys):
    parser = plot_trigger_patches.APPLICATION.build_parser()

    args = parser.parse_args(
        [
            SITE_OPTION,
            SITE_NAME,
            TELESCOPE_OPTION,
            TELESCOPE_NAME,
            MODEL_VERSION_OPTION,
            MODEL_VERSION_VALUE,
        ]
    )
    assert args.site == SITE_NAME
    assert args.telescope == TELESCOPE_NAME
    assert args.model_version == MODEL_VERSION_VALUE

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                SITE_OPTION,
                SITE_NAME,
                TELESCOPE_OPTION,
                TELESCOPE_NAME,
                MODEL_VERSION_OPTION,
                MODEL_VERSION_VALUE,
                "--photons_per_pixel",
                "500",
            ]
        )
    assert "unrecognized arguments" in capsys.readouterr().err


def test_run_writes_configuration_report_without_exporting_simulation_files(
    tmp_test_directory, monkeypatch
):
    camera = SimpleNamespace(
        telescope_name=TELESCOPE_NAME,
        pixels={
            PIXEL_IDS: [10, 11, 12],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "pixel_shape": 1,
            "pixel_diameter": 0.8,
            "rotate_angle": 0.0,
            "orientation": 0.0,
            "pix_on": [True, True, True],
        },
        get_neighbor_pixels=lambda: [[1], [0, 2], [1]],
    )
    groups = Table(rows=[(1, "analog_sum")], names=[GROUP_ID, "kind"])
    members = Table(rows=[(1, 10), (1, 11)], names=[GROUP_ID, PIXEL_ID])
    telescope = MagicMock()
    telescope.camera = camera
    telescope.name = TELESCOPE_NAME
    telescope.get_parameter_table.side_effect = [groups, members]
    monkeypatch.setattr(plot_trigger_patches, "TelescopeModel", MagicMock(return_value=telescope))
    monkeypatch.setattr(
        plot_trigger_patches,
        "read_overwrite_model_parameter_dict",
        MagicMock(return_value={}),
    )
    app_context = SimpleNamespace(
        args={
            "site": SITE_NAME,
            "telescope": TELESCOPE_NAME,
            "model_version": MODEL_VERSION_VALUE,
            "output_file": None,
            "overwrite_model_parameters": None,
        },
        paths=SimpleNamespace(
            get_output_directory=lambda: tmp_test_directory,
            get_output_file=lambda file_name: Path(tmp_test_directory) / file_name,
        ),
        model_reader=MagicMock(),
        logger=MagicMock(),
    )

    output_files = plot_trigger_patches.run(app_context)

    assert output_files["pdf_file"].is_file()
    assert not list(Path(tmp_test_directory).glob("*.png"))
    assert output_files[PIXELS_FILE_KEY].is_file()
    assert output_files[GROUPS_FILE_KEY].is_file()
    assert output_files[SUMMARY_FILE_KEY].is_file()
    summary = json.loads(output_files[SUMMARY_FILE_KEY].read_text(encoding="utf-8"))
    assert summary["runtime_trigger_status"] == "not tested"
    assert summary["membership_coverage_status"] == "review"
    assert summary["unclassified_pixel_ids"] == [12]
    assert len(summary["patch_configuration_sha256"]) == 64
    assert summary["simulation_executed"] is False
    expected_page_summary = format_trigger_patch_summary(
        Table.read(output_files[PIXELS_FILE_KEY], format=ECSV_FORMAT),
        Table.read(output_files[GROUPS_FILE_KEY], format=ECSV_FORMAT),
    )
    app_context.logger.info.assert_any_call("%s", expected_page_summary)
    telescope.model_reader.export_model_files.assert_not_called()
    assert summary["pixel_count"] == 3
