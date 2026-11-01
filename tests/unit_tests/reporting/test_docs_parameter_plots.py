import re
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable, Table

from simtools.reporting import docs_parameter_plots
from simtools.reporting.docs_read_parameters import ReadParameters


def test__convert_to_md(telescope_model_lst, tmp_test_directory, mocker, run_context):
    args = {
        "telescope": telescope_model_lst.name,
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
    }
    output_path = tmp_test_directory
    read_parameters = ReadParameters(args=args, output_path=output_path, context=run_context)
    parameter_name = "pm_photoelectron_spectrum"

    # Mock _generate_plots to avoid file I/O
    mocker.patch.object(read_parameters, "_generate_plots", return_value=[])

    # testing with invalid file
    with pytest.raises(FileNotFoundError, match="Data file not found: "):
        read_parameters._convert_to_md(parameter_name, "1.0.0", "invalid-file.dat")

    # testing with valid file
    valid_file = Path(tmp_test_directory) / "valid_parameter.dat"
    valid_lines = [f"# line {index}\n" for index in range(30)]
    valid_file.write_text("".join(valid_lines), encoding="utf-8")
    new_file = read_parameters._convert_to_md(parameter_name, "1.0.0", str(valid_file))
    assert isinstance(new_file, str)
    assert Path(output_path / new_file).exists()

    with Path(output_path / new_file).open("r", encoding="utf-8") as mdfile:
        md_content = mdfile.read()

    match = re.search(r"```\n(.*?)\n```", md_content, re.DOTALL)
    assert match, "Code block with file contents not found"

    code_block = match.group(1)
    line_count = len(code_block.strip().splitlines())
    assert line_count == 30

    # Compare to the first 30 lines of the local input file
    expected_block = "".join(valid_lines).rstrip()

    assert code_block.strip() == expected_block.strip()

    # testing with non-utf-8 file generated on the fly
    non_utf_file = Path(tmp_test_directory) / "example_non_utf-8_file.lis"
    non_utf_file.write_bytes(
        b"# Non-UTF sample file\n"
        b"# Includes invalid UTF-8 bytes to trigger Latin-1 fallback: \x80\xff\n"
        b"1 2 3\n"
    )
    new_file = read_parameters._convert_to_md(parameter_name, "1.0.0", str(non_utf_file))
    assert isinstance(new_file, str)
    assert Path(output_path / new_file).exists()


@pytest.mark.parametrize(
    ("parameter", "parameter_version", "patched_method", "return_value"),
    [
        ("some_param", "1.0.0", "_plot_parameter_tables", ["plot2"]),
        ("camera_pixel_layout", "1.0.0", "plot_camera_config", ["camera_plot"]),
        (
            "camera_trigger_groups",
            "1.0.0",
            "_plot_camera_trigger_membership",
            ["trigger_plot"],
        ),
        (
            "camera_trigger_members",
            "1.0.0",
            "_plot_camera_trigger_membership",
            ["trigger_plot"],
        ),
        ("mirror_list", "1.0.0", "plot_mirror_config", ["mirror_plot"]),
        (
            "primary_mirror_segmentation",
            "1.0.0",
            "plot_mirror_config",
            ["mirror_plot"],
        ),
        (
            "secondary_mirror_segmentation",
            "1.0.0",
            "plot_mirror_config",
            ["mirror_plot"],
        ),
        ("some_param", None, None, []),
    ],
)
def test__generate_plots(
    tmp_test_directory, parameter, parameter_version, patched_method, return_value, run_context
):
    read_parameters = ReadParameters(
        args={"telescope": "LSTN-design", "site": "North", "model_version": "6.0.0"},
        output_path=tmp_test_directory,
        context=run_context,
    )
    input_file = Path(tmp_test_directory / "dummy_param.dat")
    input_file.write_text("dummy content")

    if patched_method is None:
        assert (
            read_parameters._generate_plots(
                parameter, parameter_version, input_file, tmp_test_directory
            )
            == return_value
        )
        return

    patch_target = (
        read_parameters if patched_method == "_plot_parameter_tables" else docs_parameter_plots
    )
    with patch.object(patch_target, patched_method, return_value=return_value) as mock_plot:
        assert (
            read_parameters._generate_plots(
                parameter, parameter_version, input_file, tmp_test_directory
            )
            == return_value
        )
        if patched_method == "_plot_parameter_tables":
            mock_plot.assert_called_once_with(
                parameter, parameter_version, tmp_test_directory, model_version=None
            )
        elif patched_method == "plot_camera_config":
            mock_plot.assert_called_once_with(
                parameter_version,
                tmp_test_directory,
                "LSTN-design",
                "North",
                "6.0.0",
                read_parameters.model_reader,
                context=run_context,
            )
        elif patched_method == "_plot_camera_trigger_membership":
            mock_plot.assert_called_once_with(
                tmp_test_directory,
                "LSTN-design",
                "North",
                "6.0.0",
                read_parameters.model_reader,
                context=run_context,
            )
        elif patched_method == "plot_mirror_config":
            mock_plot.assert_called_once_with(
                parameter,
                parameter_version,
                input_file,
                tmp_test_directory,
                "LSTN-design",
                "North",
                "6.0.0",
                context=run_context,
            )


def test_plot_camera_trigger_membership_summary(tmp_test_directory, mocker, run_context):
    output_path = Path(tmp_test_directory)
    model_reader = mocker.Mock()
    model_reader.get_model_parameters.return_value = {
        "camera_pixel_layout": {"parameter": "camera_pixel_layout"},
        "camera_pixel_types": {
            "parameter": "camera_pixel_types",
            "value": [{"shape": 1, "diameter_cm": 1.0}],
        },
        "camera_rotate": {"parameter": "camera_rotate", "value": 0.0},
        "camera_trigger_groups": {"parameter": "camera_trigger_groups"},
        "camera_trigger_members": {"parameter": "camera_trigger_members"},
    }
    tables = {
        "camera_pixel_layout": QTable(
            {
                "pixel_id": [1, 2, 3],
                "x_cm": [0.0, 1.0, 0.0],
                "y_cm": [0.0, 0.0, 1.0],
                "enabled": [True, True, True],
            }
        ),
        "camera_trigger_groups": Table({"group_id": [10, 11], "kind": ["L1", "L1"]}),
        "camera_trigger_members": Table({"group_id": [10, 10, 11, 11], "pixel_id": [1, 2, 2, 3]}),
    }
    tables["camera_pixel_layout"]["x_cm"].unit = u.cm
    tables["camera_pixel_layout"]["y_cm"].unit = u.cm
    model_reader.get_parameter_table.side_effect = lambda parameter: tables[parameter["parameter"]]
    plots = docs_parameter_plots._plot_camera_trigger_membership(
        output_path,
        "LSTN-design",
        "North",
        "8.0.0",
        model_reader,
        context=run_context,
    )

    assert len(plots) == 1
    assert plots[0].startswith("camera_trigger_membership_LSTN-design_")
    assert (output_path / f"{plots[0]}.png").is_file()

    member_page_plots = docs_parameter_plots._plot_camera_trigger_membership(
        output_path,
        "LSTN-design",
        "North",
        "8.0.0",
        model_reader,
        context=run_context,
    )
    assert member_page_plots == plots


def test_count_trigger_groups_per_pixel_counts_unique_memberships():
    groups = Table({"group_id": [10, 11]})
    members = Table({"group_id": [10, 10, 11, 11, 11], "pixel_id": [1, 2, 2, 3, 2]})

    counts = docs_parameter_plots._count_trigger_groups_per_pixel([1, 2, 3, 4], groups, members)

    assert counts.tolist() == [1, 2, 1, 0]


def test_count_trigger_groups_per_pixel_rejects_unknown_member_pixel():
    groups = Table({"group_id": [10]})
    members = Table({"group_id": [10], "pixel_id": [99]})

    with pytest.raises(ValueError, match="Trigger member pixel 99 is not in the camera"):
        docs_parameter_plots._count_trigger_groups_per_pixel([1, 2], groups, members)


@pytest.mark.parametrize(
    ("pixel_ids", "groups", "members", "error"),
    [
        ([1, 1], {"group_id": [10]}, {"group_id": [10], "pixel_id": [1]}, "pixel IDs"),
        ([1], {"group_id": [10, 10]}, {"group_id": [10], "pixel_id": [1]}, "group IDs"),
        ([1], {"group_id": []}, {"group_id": [], "pixel_id": []}, "no trigger groups"),
        ([1], {"group_id": [10]}, {"group_id": [11], "pixel_id": [1]}, "unknown trigger group"),
        ([1], {"group_id": [10, 11]}, {"group_id": [10], "pixel_id": [1]}, "no members"),
        (
            [1],
            {"id": [10]},
            {"group_id": [10], "pixel_id": [1]},
            "missing required column group_id",
        ),
    ],
)
def test_count_trigger_groups_per_pixel_rejects_invalid_configuration(
    pixel_ids, groups, members, error
):
    with pytest.raises(ValueError, match=error):
        docs_parameter_plots._count_trigger_groups_per_pixel(
            pixel_ids, Table(groups), Table(members)
        )


@pytest.mark.parametrize(
    ("telescope", "site", "model_version", "parameters"),
    [
        (None, "North", "8.0.0", {}),
        ("LSTN-design", "North", "8.0.0", {}),
        (
            "LSTN-design",
            "North",
            "8.0.0",
            {"camera_pixel_layout": {"parameter": "camera_pixel_layout"}},
        ),
    ],
)
def test_plot_camera_trigger_membership_skips_missing_context_or_parameters(
    tmp_test_directory, mocker, run_context, telescope, site, model_version, parameters
):
    model_reader = mocker.Mock()
    model_reader.get_model_parameters.return_value = parameters

    result = docs_parameter_plots._plot_camera_trigger_membership(
        Path(tmp_test_directory), telescope, site, model_version, model_reader, run_context
    )

    assert result == []
    if telescope is None:
        model_reader.get_model_parameters.assert_not_called()
    else:
        model_reader.get_parameter_table.assert_not_called()


def test__plot_parameter_tables(tmp_test_directory, mocker, run_context):
    from pathlib import Path

    from simtools.visualization import plot_tables

    args = {"telescope": "LSTN-design", "site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(args=args, output_path=tmp_test_directory, context=run_context)

    # Mock plot_tables.generate_plot_configurations to return (configs, output_files) tuple
    mock_config = {
        "tables": [{"table_name": "test_table", "column_x": "x", "column_y": "y"}],
        "title": "Test table",
        "xtitle": "x",
        "ytitle": "y",
        "plot_name": "pm_photoelectron_spectrum_1.0.0_North_LSTN-design",
    }
    mock_files = [Path("pm_photoelectron_spectrum_1.0.0_North_LSTN-design.png")]
    mocker.patch.object(
        plot_tables, "generate_plot_configurations", return_value=([mock_config], mock_files)
    )

    # Mock plot_tables.plot to avoid actual plotting
    mocker.patch.object(
        plot_tables,
        "read_table_data",
        return_value={
            "test_table": np.array([(1.0, 2.0), (2.0, 3.0)], dtype=[("x", float), ("y", float)])
        },
    )
    mocker.patch.object(docs_parameter_plots, "_get_figure_cache_key", return_value="cached-key")

    result = read_parameters._plot_parameter_tables(
        "pm_photoelectron_spectrum", "1.0.0", Path(tmp_test_directory)
    )
    assert result == ["parameter_plot_cached-key"]
    assert Path(tmp_test_directory, f"{result[0]}.png").is_file()

    args = {"telescope": None, "site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(args=args, output_path=tmp_test_directory, context=run_context)

    # Mock to return None (empty)
    mocker.patch.object(plot_tables, "generate_plot_configurations", return_value=None)

    result = read_parameters._plot_parameter_tables(
        "camera_pixel_layout", "1.0.0", Path(tmp_test_directory)
    )
    assert result == []


def test_table_plot_cache_key_uses_rendered_data_and_visible_settings(run_context):
    data = {"sample": np.array([(1.0, 2.0)], dtype=[("x", float), ("y", float)])}
    config = {
        "title": "Sample",
        "xtitle": "x",
        "ytitle": "y",
        "legend_location": "best",
        "xlim": [0.0, 2.0],
        "tables": [{"parameter": "example", "site": "North", "model_version": "8.0.0"}],
    }
    key = docs_parameter_plots._get_figure_cache_key(config, data, run_context)

    assert (
        docs_parameter_plots._get_figure_cache_key(
            {**config, "tables": [{"parameter": "example", "site": "South"}]}, data, run_context
        )
        == key
    )
    assert (
        docs_parameter_plots._get_figure_cache_key(
            {**config, "title": "Different title"}, data, run_context
        )
        != key
    )
    changed_data = {"sample": np.array([(1.0, 3.0)], dtype=[("x", float), ("y", float)])}
    assert docs_parameter_plots._get_figure_cache_key(config, changed_data, run_context) != key


def test_parameter_plot_overlay_uses_latest_three_compatible_versions(
    tmp_test_directory,
):
    all_versions = {
        "6.0.0": {"fadc_pulse_shape": {"parameter_version": "1.0.0"}},
        "7.0.0": {"fadc_pulse_shape": {"parameter_version": "2.0.0"}},
        "8.0.0": {"fadc_pulse_shape": {"parameter_version": "3.0.0"}},
        "9.0.0": {"fadc_pulse_shape": {"parameter_version": "4.0.0"}},
    }

    assert docs_parameter_plots._parameter_plot_candidates(
        all_versions, "fadc_pulse_shape", "3.0.0", "8.0.0"
    ) == [("1.0.0", "6.0.0"), ("2.0.0", "7.0.0"), ("3.0.0", "8.0.0")]


def test_parameter_plot_candidates_skips_missing_versions_and_newer_models():
    all_versions = {
        "6.0.0": {"pulse_shape": {"parameter_version": "1.0.0"}},
        "7.0.0": {"pulse_shape": None},
        "8.0.0": {"pulse_shape": {"parameter_version": "3.0.0"}},
        "9.0.0": {"pulse_shape": {"parameter_version": "4.0.0"}},
    }

    assert docs_parameter_plots._parameter_plot_candidates(
        all_versions, "pulse_shape", "3.0.0", "8.0.0"
    ) == [("1.0.0", "6.0.0"), ("3.0.0", "8.0.0")]


def test_matching_version_plot_config_requires_same_signature_and_axis_units(run_context, mocker):
    current = {
        "title": "Pulse shape",
        "tables": [
            {
                "parameter": "pulse_shape",
                "site": "North",
                "model_version": "8.0.0",
                "column_x": "x",
                "column_y": "y",
            }
        ],
    }
    different_plot = {**current, "title": "Other plot"}
    multiple_tables = {**current, "tables": [*current["tables"], current["tables"][0]]}
    different_units = {
        **current,
        "tables": [{**current["tables"][0], "model_version": "7.0.0"}],
    }
    matching = {
        **current,
        "tables": [{**current["tables"][0], "model_version": "6.0.0"}],
    }
    model_reader = mocker.Mock()

    def export_table(model_version, **_kwargs):
        x_unit = u.cm if model_version == "7.0.0" else u.ns
        return QTable({"x": [1.0] * x_unit, "y": [0.5] * u.dimensionless_unscaled})

    model_reader.export_model_file.side_effect = export_table
    result = docs_parameter_plots._matching_version_plot_config(
        [different_plot, multiple_tables, different_units, matching],
        docs_parameter_plots._plot_signature(current),
        current["tables"][0],
        model_reader,
        run_context,
    )

    assert result == matching
    assert model_reader.export_model_file.call_count == 4


def test_table_axis_units_returns_none_when_export_or_columns_are_unavailable(run_context, mocker):
    table_config = {"parameter": "pulse_shape", "site": "North", "column_x": "x", "column_y": "y"}
    model_reader = mocker.Mock()
    model_reader.export_model_file.side_effect = ValueError("table export unavailable")
    assert docs_parameter_plots._table_axis_units(table_config, model_reader, run_context) is None

    model_reader.export_model_file.side_effect = None
    model_reader.export_model_file.return_value = QTable({"x": [1.0] * u.ns})
    assert docs_parameter_plots._table_axis_units(table_config, model_reader, run_context) is None


def test_read_parameter_plot_version_reuses_current_data_for_current_version(run_context, mocker):
    data = {"series": np.array([(1.0, 2.0)], dtype=[("x", float), ("y", float)])}
    model_reader = mocker.Mock()

    result = docs_parameter_plots._read_parameter_plot_version(
        "pulse_shape",
        "3.0.0",
        "3.0.0",
        "8.0.0",
        {"tables": [{}]},
        data,
        (),
        "North",
        "LSTN-design",
        Path(),
        model_reader,
        run_context,
    )

    assert result is data
    model_reader.assert_not_called()


def test_read_parameter_plot_version_reads_older_schema_compatible_table(
    tmp_test_directory, run_context, mocker
):
    table_config = {
        "parameter": "fadc_pulse_shape",
        "site": "North",
        "telescope": "LSTN-design",
        "parameter_version": "1.0.0",
        "model_version": "8.0.0",
        "column_x": "time",
        "column_y": "amplitude",
    }
    plot_config = {
        "type": "fadc_pulse_shape",
        "title": "FADC pulse shape",
        "xtitle": "Time [ns]",
        "ytitle": "Amplitude [a.u.]",
        "xscale": "linear",
        "yscale": "linear",
        "no_markers": True,
        "tables": [table_config],
    }
    current_data = {
        "time vs amplitude": np.array([(0.0, 1.0)], dtype=[("time", float), ("amplitude", float)])
    }
    table = QTable({"time": [0.0, 1.0] * u.ns, "amplitude": [1.0, 0.5]})
    model_reader = mocker.Mock()
    model_reader.get_model_parameter.return_value = {
        "fadc_pulse_shape": {"model_parameter_schema_version": "0.2.0"}
    }
    model_reader.export_model_file.return_value = table

    result = docs_parameter_plots._read_parameter_plot_version(
        "fadc_pulse_shape",
        "2.0.0",
        "1.0.0",
        "7.0.0",
        plot_config,
        current_data,
        docs_parameter_plots._plot_signature(plot_config),
        "North",
        "LSTN-design",
        Path(tmp_test_directory),
        model_reader,
        run_context,
    )

    assert len(result) == 1
    np.testing.assert_allclose(result["time vs amplitude"]["amplitude"], [1.0, 0.5])


def test_read_parameter_plot_version_skips_parameter_without_plot_configuration(
    tmp_test_directory, run_context, mocker
):
    model_reader = mocker.Mock()
    model_reader.get_model_parameter.return_value = {
        "num_gains": {"model_parameter_schema_version": "0.1.0"}
    }

    result = docs_parameter_plots._read_parameter_plot_version(
        "num_gains",
        "2.0.0",
        "1.0.0",
        "7.0.0",
        {"tables": [{}]},
        {},
        (),
        "North",
        "LSTN-design",
        Path(tmp_test_directory),
        model_reader,
        run_context,
    )

    assert result is None
    model_reader.export_model_file.assert_not_called()


def test_read_parameter_plot_version_skips_incompatible_plot_configuration(
    tmp_test_directory, run_context, mocker
):
    model_reader = mocker.Mock()
    model_reader.get_model_parameter.return_value = {
        "fadc_pulse_shape": {"model_parameter_schema_version": "0.2.0"}
    }
    model_reader.export_model_file.return_value = QTable(
        {"time": [0.0, 1.0] * u.ns, "amplitude": [1.0, 0.5]}
    )

    result = docs_parameter_plots._read_parameter_plot_version(
        "fadc_pulse_shape",
        "2.0.0",
        "1.0.0",
        "7.0.0",
        {"tables": [{"parameter": "fadc_pulse_shape", "site": "North"}]},
        {},
        ("different",),
        "North",
        "LSTN-design",
        Path(tmp_test_directory),
        model_reader,
        run_context,
    )

    assert result is None
    model_reader.export_model_file.assert_called_once()


@pytest.mark.parametrize(
    ("telescope", "model_version", "plot_config", "current_data"),
    [
        (None, "8.0.0", {"tables": [{}]}, {"series": [1]}),
        ("LSTN-design", None, {"tables": [{}]}, {"series": [1]}),
        ("LSTN-design", "8.0.0", {"tables": [{}, {}]}, {"series": [1]}),
        ("LSTN-design", "8.0.0", {"tables": [{}]}, {"first": [1], "second": [2]}),
    ],
)
def test_compatible_parameter_plot_versions_skips_ineligible_overlay(
    tmp_test_directory, run_context, telescope, model_version, plot_config, current_data
):
    read_parameters = ReadParameters(
        args={"site": "North", "telescope": "LSTN-design", "model_version": "8.0.0"},
        output_path=tmp_test_directory,
        context=run_context,
    )

    result = read_parameters._get_compatible_parameter_plot_versions(
        "pulse_shape",
        "1.0.0",
        plot_config,
        current_data,
        "North",
        telescope,
        model_version,
        Path(tmp_test_directory),
    )

    assert result == []


def test_matching_version_plot_config_returns_none_without_matching_units(run_context, mocker):
    config = {
        "title": "Pulse shape",
        "tables": [{"parameter": "pulse_shape", "site": "North", "column_x": "x", "column_y": "y"}],
    }
    model_reader = mocker.Mock()
    model_reader.export_model_file.return_value = QTable(
        {"x": [1.0] * u.ns, "y": [0.5] * u.dimensionless_unscaled}
    )

    result = docs_parameter_plots._matching_version_plot_config(
        [config],
        docs_parameter_plots._plot_signature({**config, "title": "Different"}),
        config["tables"][0],
        model_reader,
        run_context,
    )

    assert result is None


def test__plot_camera_components_reuses_resolved_camera_data(
    tmp_test_directory, mocker, run_context
):
    read_parameters = ReadParameters(
        args={"telescope": "LSTN-01", "site": "North", "model_version": "6.0.0"},
        output_path=tmp_test_directory,
        context=run_context,
    )
    input_file = Path(tmp_test_directory / "camera_pixel_layout.ecsv")
    input_file.touch()
    mocker.patch.object(read_parameters.model_reader, "get_model_parameters", return_value={})
    mocker.patch.object(
        read_parameters.model_reader,
        "get_model_parameter",
        return_value={"camera_pixel_layout": {"value": "pixels"}},
    )
    mocker.patch(
        "simtools.reporting.model_figures.resolve_camera_components",
        return_value={
            "pixels": [{"pixel_id": 1, "x_cm": 1.0, "y_cm": 2.0}],
            "pixel_types": [{"shape": 1, "diameter_cm": 1.0}],
            "rotate": 0.0,
        },
    )
    mocker.patch("simtools.reporting.model_figures.camera_model.Camera.from_configuration")
    mocker.patch("simtools.reporting.model_figures.plot_pixels.raw_pixels_from_configuration")

    def save_image(_config=None, output_file=None, **_kwargs):
        output_path = Path(output_file).with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()

    mock_plot = mocker.patch("simtools.reporting.model_figures.plot_pixels.plot")
    mock_plot.side_effect = save_image
    first = read_parameters._generate_plots(
        "camera_pixel_layout", "1.0.0", input_file, tmp_test_directory, model_version="1.0.0"
    )
    second = read_parameters._generate_plots(
        "camera_pixel_layout", "1.0.0", input_file, tmp_test_directory, model_version="7.0.0"
    )

    assert first == second
    assert first[0].startswith("camera_pixel_layout_")
    mock_plot.assert_called_once()


def test_write_file_flag_section_without_generated_plot(tmp_test_directory, run_context):
    parameter = "param_x"
    rp = ReadParameters(
        args={"telescope": "LSTN-01", "site": "North", "model_version": "6.0.0"},
        output_path=tmp_test_directory,
        context=run_context,
    )
    rp._get_telescope_identifier = lambda mv=None: "LSTN-01"

    comparison_data = {
        parameter: [
            {
                "parameter_version": "1.0.0",
                "model_version": "6.0.0",
                "file_flag": True,
            }
        ]
    }

    buf = StringIO()
    with patch.object(rp.model_reader, "get_model_parameters", return_value={}):
        rp._write_file_flag_section(buf, parameter, comparison_data, "telescopes")
    content = buf.getvalue()

    assert content == ""


def test_write_file_flag_section_lists_generated_plots_for_matching_version(
    tmp_test_directory, run_context
):
    image_dir = Path(tmp_test_directory) / "_images"
    image_dir.mkdir()
    (image_dir / "param_x_1.0.0_North_LSTN-01_a.png").touch()
    (image_dir / "param_x_1.0.0_North_LSTN-01_b.png").touch()
    (image_dir / "param_x_2.0.0_North_LSTN-01.png").touch()
    rp = ReadParameters(
        args={"telescope": "LSTN-01", "site": "North", "model_version": "6.0.0"},
        output_path=tmp_test_directory,
        context=run_context,
    )
    rp.context.paths.output_path["default"] = Path(tmp_test_directory)
    rp._get_telescope_identifier = lambda mv=None: "LSTN-01"
    comparison_data = {
        "param_x": [
            {
                "parameter_version": "1.0.0",
                "model_version": "6.0.0",
                "file_flag": True,
            },
            {
                "parameter_version": "2.0.0",
                "model_version": "5.0.0",
                "file_flag": True,
            },
        ]
    }

    buf = StringIO()
    with (
        patch.object(
            rp.model_reader,
            "get_model_parameters",
            return_value={
                "param_x": {
                    "parameter_version": "2.0.0",
                    "value": "data.ecsv",
                    "file": True,
                    "instrument": "LSTN-01",
                }
            },
        ) as get_params,
        patch.object(rp.model_reader, "export_model_files"),
        patch.object(
            rp,
            "_generate_plots",
            return_value=[
                "param_x_2.0.0_North_LSTN-01_a",
                "param_x_2.0.0_North_LSTN-01_b",
            ],
        ) as generate_plots,
    ):
        get_params.return_value["param_x"]["parameter_version"] = "2.0.0"
        (image_dir / "param_x_2.0.0_North_LSTN-01_a.png").touch()
        (image_dir / "param_x_2.0.0_North_LSTN-01_b.png").touch()
        rp._write_file_flag_section(buf, "param_x", comparison_data, "telescopes")

    content = buf.getvalue()
    get_params.assert_called_once_with(
        site="North",
        array_element_name="LSTN-01",
        collection="telescopes",
        model_version="5.0.0",
    )
    assert "param_x_2.0.0_North_LSTN-01_a.png" in content
    assert "param_x_1.0.0_North_LSTN-01_a.png" not in content
    assert generate_plots.call_args.args[2] == (
        Path(tmp_test_directory) / "model" / "data-LSTN-01.ecsv"
    )


def test__convert_to_md_writes_plot_references(tmp_path, mocker, run_context):
    rp = ReadParameters(args={}, output_path=tmp_path, context=run_context)
    input_file = tmp_path / "parameter_data.dat"
    input_file.write_text("\n".join(f"line {i}" for i in range(40)), encoding="utf-8")

    mocker.patch.object(rp, "_generate_plots", return_value=["plot_a", "plot_b"])
    relative_md = rp._convert_to_md("param", "1.0.0", input_file)

    content = (tmp_path / relative_md).read_text(encoding="utf-8")
    assert "![Parameter plot.]" in content
    assert "plot_a.png" in content
    assert "plot_b.png" in content


@pytest.mark.parametrize(
    ("parameter_version", "preexisting_plot"),
    [(None, False), ("1.0.0", False), ("1.0.0", True)],
)
def test__plot_mirror_config(
    tmp_test_directory, mocker, parameter_version, preexisting_plot, run_context
):
    read_parameters = ReadParameters(
        args={"telescope": "LSTN-01", "site": "North", "model_version": "6.0.0"},
        output_path=tmp_test_directory,
        context=run_context,
    )
    input_file = Path(tmp_test_directory / "mirror_config.ecsv")
    Table(
        {
            "mirror_panel_id": [0],
            "mirror_x": [0.0] * u.cm,
            "mirror_y": [0.0] * u.cm,
            "mirror_diameter": [100.0] * u.cm,
            "focal_length": [1000.0] * u.cm,
            "shape_type": [1],
        }
    ).write(input_file, format="ascii.ecsv")
    mock_plot = mocker.patch("simtools.reporting.model_figures.plot_mirrors.plot")
    result = read_parameters._generate_plots(
        "mirror_list", parameter_version, input_file, tmp_test_directory
    )

    assert result == [] if parameter_version is None else len(result) == 1
    if result:
        assert result[0].startswith("mirror_config_")
    if parameter_version is None:
        mock_plot.assert_not_called()
    else:
        if preexisting_plot:
            # The content-addressed output path is determined by the figure inputs.
            mock_plot.assert_called_once()
        else:
            mock_plot.assert_called_once_with(
                config={
                    "parameter": "mirror_list",
                    "telescope": "LSTN-01",
                    "parameter_version": "1.0.0",
                    "site": "North",
                    "model_version": "6.0.0",
                },
                output_file=Path(tmp_test_directory / result[0]),
                data_file_path=input_file,
                mirrors=mock_plot.call_args.kwargs["mirrors"],
            )
