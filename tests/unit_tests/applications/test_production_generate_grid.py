from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import simtools.applications.plot_array_layout as plot_array_layout_app
import simtools.applications.production_generate_grid as app
from simtools.configuration.commandline_parser import CommandLineParser


def _parser():
    parser = CommandLineParser()
    parser.add_argument_definitions(app._GRID_ARGUMENTS)
    return parser


def _full_parser():
    return app.APPLICATION.build_parser()


@patch("simtools.applications.production_generate_grid.generate_job_grid")
@patch("simtools.applications.production_generate_grid.MetadataCollector")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_generates_job_grid(mock_start, mock_metadata_collector, mock_generate_job_grid):
    io_handler = Mock()
    io_handler.get_output_file.return_value = Path("job_grid.ecsv")
    args = {
        "output_file": "job_grid.ecsv",
        "run_number_offset": 10,
    }
    mock_start.return_value = SimpleNamespace(args=args, io_handler=io_handler)
    metadata = {"cta": {"activity": {"name": "production_generate_grid"}}}
    mock_metadata_collector.return_value.get_top_level_metadata.return_value = metadata
    app.main()

    mock_metadata_collector.assert_called_once_with(args)
    mock_generate_job_grid.assert_called_once_with(args, Path("job_grid.ecsv"), metadata=metadata)
    mock_start.assert_called_once_with()


def test_full_parser_retains_supported_shared_arguments():
    parser = _full_parser()
    actions = {action.dest: action for action in parser._actions}

    expected = {
        "array_layout_name",
        "azimuth_angle",
        "correct_for_b_field_alignment",
        "core_scatter",
        "curved_atmosphere_min_zenith_angle",
        "energy_range",
        "eslope",
        "event_number_first_shower",
        "model_version",
        "output_file",
        "output_path",
        "overwrite_model_parameters",
        "primary",
        "primary_id_type",
        "run_number_offset",
        "run_number",
        "showers_per_run",
        "show_options",
        "site",
        "view_cone",
        "zenith_angle",
    }
    assert expected <= set(actions)
    assert "array_element_list" not in actions
    assert actions["output_file"].default == "job_grid.ecsv"
    assert actions["output_file"].help == "Output ECSV production job grid."


def test_plot_array_layout_parser_retains_ignore_missing_design_model():
    actions = {action.dest for action in plot_array_layout_app.APPLICATION.build_parser()._actions}

    assert "ignore_missing_design_model" in actions


def test_plot_array_layout_parser_accepts_parameter_file_layout_selector():
    args = plot_array_layout_app.APPLICATION.build_parser().parse_args(
        [
            "--array_layout_parameter_file",
            "array_layouts.json",
            "--array_layout_name_from_parameter_file",
            "CTAO-South-2-MSTs-5-SSTs",
            "--model_version",
            "7.0.0",
        ]
    )

    assert args.array_layout_parameter_file == "array_layouts.json"
    assert args.array_layout_name_from_parameter_file == ["CTAO-South-2-MSTs-5-SSTs"]


def test_full_parser_accepts_minimum_direct_configuration():
    args = _full_parser().parse_args(
        [
            "--model_version",
            "7.0.0",
            "--site",
            "North",
            "--array_layout_name",
            "LSTN-01",
            "--primary",
            "gamma",
            "--showers_per_run",
            "1000",
        ]
    )

    assert args.model_version == ["7.0.0"]
    assert args.site == "North"
    assert args.array_layout_name == ["LSTN-01"]
    assert not hasattr(args, "telescope")
    assert args.output_file == "job_grid.ecsv"


def test_application_parse_allows_show_options_without_required_runtime_arguments(
    monkeypatch, capsys
):
    monkeypatch.setattr("sys.argv", ["production_generate_grid.py", "--show_options", "site"])

    with pytest.raises(SystemExit) as exc:
        app.APPLICATION._parse()

    assert exc.value.code == 0
    assert "Available values:" in capsys.readouterr().out
