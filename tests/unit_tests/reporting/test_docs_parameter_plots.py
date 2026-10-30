import re
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import astropy.units as u
import pytest
from astropy.table import Table

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


def test__plot_parameter_tables(tmp_test_directory, mocker, run_context):
    from pathlib import Path

    from simtools.visualization import plot_tables

    args = {"telescope": "LSTN-design", "site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(args=args, output_path=tmp_test_directory, context=run_context)

    # Mock plot_tables.generate_plot_configurations to return (configs, output_files) tuple
    mock_config = {
        "tables": [{"table_name": "test_table"}],
        "plot_name": "pm_photoelectron_spectrum_1.0.0_North_LSTN-design",
    }
    mock_files = [Path("pm_photoelectron_spectrum_1.0.0_North_LSTN-design.png")]
    mocker.patch.object(
        plot_tables, "generate_plot_configurations", return_value=([mock_config], mock_files)
    )

    # Mock plot_tables.plot to avoid actual plotting
    mocker.patch.object(plot_tables, "read_table_data", return_value={"test_table": []})
    mocker.patch.object(docs_parameter_plots, "_get_figure_cache_key", return_value="cached-key")
    mock_render = mocker.patch.object(plot_tables, "_plot_data")

    result = read_parameters._plot_parameter_tables(
        "pm_photoelectron_spectrum", "1.0.0", Path(tmp_test_directory)
    )
    assert result == ["pm_photoelectron_spectrum_1-0-0_North_LSTN-design_cached-key"]
    mock_render.assert_called_once()

    args = {"telescope": None, "site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(args=args, output_path=tmp_test_directory, context=run_context)

    # Mock to return None (empty)
    mocker.patch.object(plot_tables, "generate_plot_configurations", return_value=None)

    result = read_parameters._plot_parameter_tables(
        "camera_pixel_layout", "1.0.0", Path(tmp_test_directory)
    )
    assert result == []


def test__plot_parameter_tables_reuses_only_identical_rendered_inputs(
    tmp_path, mocker, run_context
):
    from pathlib import Path

    from simtools.visualization import plot_tables

    read_parameters = ReadParameters(
        args={"telescope": "LSTN-design", "site": "North", "model_version": "8.0.0"},
        output_path=tmp_path,
        context=run_context,
    )
    mocker.patch.object(read_parameters, "_get_telescope_identifier", return_value="LSTN-design")
    image_base = Path("mirror_reflectivity_1.0.0_North_LSTN-design.pdf")
    configs = [
        {"tables": [{"model_version": version, "parameter": "mirror_reflectivity"}]}
        for version in ("7.0.0", "8.0.0", "8.0.0")
    ]
    mocker.patch.object(
        plot_tables,
        "generate_plot_configurations",
        side_effect=[([config], [image_base]) for config in configs],
    )
    mocker.patch.object(
        plot_tables,
        "read_table_data",
        side_effect=[
            {"Mirror": [(300.0, 0.91)]},
            {"Mirror": [(300.0, 0.91)]},
            {"Mirror": [(300.0, 0.89)]},
        ],
    )

    def save_figure(_data, _config, output_file, *, context=None):
        Path(output_file).with_suffix(".png").touch()

    render = mocker.patch.object(plot_tables, "_plot_data", side_effect=save_figure)
    first = read_parameters._plot_parameter_tables(
        "mirror_reflectivity", "1.0.0", tmp_path, model_version="7.0.0"
    )
    second = read_parameters._plot_parameter_tables(
        "mirror_reflectivity", "1.0.0", tmp_path, model_version="8.0.0"
    )
    third = read_parameters._plot_parameter_tables(
        "mirror_reflectivity", "1.0.0", tmp_path, model_version="8.0.0"
    )

    assert first == second
    assert third != second
    assert (tmp_path / f"{first[0]}.png").is_file()
    assert (tmp_path / f"{third[0]}.png").is_file()
    assert render.call_count == 2


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
        return_value={"pixels": [(1.0, 2.0)]},
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
    assert first[0].startswith("camera_pixel_layout_LSTN-01_")
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
        assert result[0].startswith("mirror_list_LSTN-01_")
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
