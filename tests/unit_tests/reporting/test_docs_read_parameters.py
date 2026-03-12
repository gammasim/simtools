import re
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest

from simtools.reporting.docs_read_parameters import ReadParameters
from simtools.utils import names

# Test constants
QE_FILE_NAME = "qe_lst1_20200318_high+low.dat"
DESCRIPTION = "Test parameter"
SHORT_DESC = "Short"


def test_get_all_parameter_descriptions(telescope_model_lst, tmp_path):
    args = {
        "telescope": telescope_model_lst.name,
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
    }
    read_parameters = ReadParameters(args=args, output_path=tmp_path)
    # Call get_all_parameter_descriptions
    description_dict = read_parameters.get_all_parameter_descriptions()

    assert isinstance(description_dict.get("focal_length"), dict)
    assert isinstance(description_dict.get("focal_length").get("description"), str)
    assert isinstance(description_dict.get("focal_length").get("short_description"), str)
    assert isinstance(description_dict.get("focal_length").get("inst_class"), str)


def test_produce_array_element_report(telescope_model_lst, tmp_path):
    site = telescope_model_lst.site
    # Observatory branch
    rp = ReadParameters(
        args={"site": site, "model_version": "6.0.0", "observatory": True}, output_path=tmp_path
    )
    with patch.object(
        rp.db, "get_model_parameters", return_value={"site_elevation": {"value": 2200, "unit": "m"}}
    ):
        rp.produce_array_element_report()
    assert (tmp_path / f"OBS-{site}.md").exists()

    # Telescope branch
    rp2 = ReadParameters(
        args={"site": site, "model_version": "6.0.0", "telescope": telescope_model_lst.name},
        output_path=tmp_path,
    )
    with (
        patch.object(
            rp2.db,
            "get_model_parameters",
            return_value={
                "focal_length": {
                    "value": 2800.0,
                    "unit": "cm",
                    "parameter_version": "1.0.0",
                    "instrument": telescope_model_lst.name,
                }
            },
        ),
        patch.object(rp2.db, "export_model_files"),
        patch(
            "simtools.reporting.docs_read_parameters.TelescopeModel",
            return_value=DummyTelescope(
                site=site,
                name=telescope_model_lst.name,
                model_version="6.0.0",
                param_versions={"focal_length": "1.0.0"},
                design_model=telescope_model_lst.name,
            ),
        ),
    ):
        rp2.produce_array_element_report()
    assert (tmp_path / f"{telescope_model_lst.name}.md").exists()


def test_produce_model_parameter_reports(tmp_test_directory, mocker):
    args = {"site": "North", "telescope": "LSTN-01"}
    output_path = tmp_test_directory
    read_parameters = ReadParameters(args=args, output_path=output_path)

    # Mock get_model_parameters_for_all_model_versions to return quantum_efficiency data
    mock_data = {
        "6.0.2": {
            "quantum_efficiency": {
                "value": "qe_file.dat",
                "parameter_version": "1.0.0",
                "file": True,
                "type": "str",
                "instrument": "LSTN-01",  # Must match telescope
            }
        }
    }
    mocker.patch.object(
        read_parameters.db,
        "get_model_parameters_for_all_model_versions",
        return_value=mock_data,
    )

    read_parameters.produce_model_parameter_reports()

    file_path = output_path / args["telescope"] / "quantum_efficiency.md"
    assert file_path.exists()


@pytest.mark.parametrize(
    (
        "args",
        "report_method",
        "report_filename",
        "base_params",
        "current_params",
        "expected_link",
        "expected_text",
        "patch_telescope_model",
    ),
    [
        (
            {"site": "North", "telescope": "LSTN-01", "model_version": "6.0.1"},
            "produce_array_element_report",
            "LSTN-01.md",
            {
                "focal_length": {
                    "value": 2800.0,
                    "unit": "cm",
                    "parameter_version": "1.0.0",
                    "instrument": "LSTN-01",
                }
            },
            {
                "focal_length": {
                    "value": 2810.0,
                    "unit": "cm",
                    "parameter_version": "1.0.1",
                    "instrument": "LSTN-01",
                }
            },
            "../6.0.0/LSTN-01.md",
            "focal_length",
            True,
        ),
        (
            {"site": "South", "model_version": "6.2.1", "observatory": True},
            "produce_observatory_report",
            "OBS-South.md",
            {
                "site_elevation": {
                    "value": 2200,
                    "unit": "m",
                    "parameter_version": "1.0.0",
                }
            },
            {
                "site_elevation": {
                    "value": 2201,
                    "unit": "m",
                    "parameter_version": "1.0.1",
                }
            },
            "../6.2.0/OBS-South.md",
            "delta report",
            False,
        ),
    ],
)
def test_patch_version_reports_write_delta(
    tmp_test_directory,
    mocker,
    args,
    report_method,
    report_filename,
    base_params,
    current_params,
    expected_link,
    expected_text,
    patch_telescope_model,
):
    read_parameters = ReadParameters(args=args, output_path=tmp_test_directory)
    get_model_parameters = mocker.patch.object(
        read_parameters.db,
        "get_model_parameters",
        side_effect=[base_params, current_params],
    )
    export_model_files = mocker.patch.object(read_parameters.db, "export_model_files")
    telescope_model = None
    if patch_telescope_model:
        telescope_model = mocker.patch("simtools.reporting.docs_read_parameters.TelescopeModel")

    getattr(read_parameters, report_method)()

    export_model_files.assert_not_called()
    if telescope_model is not None:
        telescope_model.assert_not_called()

    content = (tmp_test_directory / report_filename).read_text(encoding="utf-8")
    assert expected_link in content
    assert expected_text in content
    assert get_model_parameters.call_count == 2


def test__convert_to_md(telescope_model_lst, tmp_test_directory, mocker):
    args = {
        "telescope": telescope_model_lst.name,
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
    }
    output_path = tmp_test_directory
    read_parameters = ReadParameters(args=args, output_path=output_path)
    parameter_name = "pm_photoelectron_spectrum"

    # Mock _generate_plots to avoid file I/O
    mocker.patch.object(read_parameters, "_generate_plots", return_value=[])

    # testing with invalid file
    with pytest.raises(FileNotFoundError, match="Data file not found: "):
        read_parameters._convert_to_md(parameter_name, "1.0.0", "invalid-file.dat")

    # testing with valid file
    valid_file = Path("tests/resources/spe_LST_2022-04-27_AP2.0e-4.dat")
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

    # Compare to actual first 30 lines of input file
    with valid_file.open("r", encoding="utf-8") as original_file:
        expected_lines = original_file.read().splitlines()[:30]
        expected_block = "\n".join(expected_lines)

    assert code_block.strip() == expected_block.strip()

    # testing with non-utf-8 file
    non_utf_file = Path("tests/resources/example_non_utf-8_file.lis")
    new_file = read_parameters._convert_to_md(parameter_name, "1.0.0", str(non_utf_file))
    assert isinstance(new_file, str)
    assert Path(output_path / new_file).exists()


@pytest.mark.parametrize(
    ("parameter", "parameter_version", "patched_method", "return_value"),
    [
        ("some_param", "1.0.0", "_plot_parameter_tables", ["plot2"]),
        ("camera_config_file", "1.0.0", "_plot_camera_config", ["camera_plot"]),
        ("mirror_list", "1.0.0", "_plot_mirror_config", ["mirror_plot"]),
        (
            "primary_mirror_segmentation",
            "1.0.0",
            "_plot_mirror_config",
            ["mirror_plot"],
        ),
        (
            "secondary_mirror_segmentation",
            "1.0.0",
            "_plot_mirror_config",
            ["mirror_plot"],
        ),
        ("some_param", None, None, []),
    ],
)
def test__generate_plots(
    tmp_test_directory, parameter, parameter_version, patched_method, return_value
):
    read_parameters = ReadParameters(
        args={"telescope": "LSTN-design", "site": "North", "model_version": "6.0.0"},
        output_path=tmp_test_directory,
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

    with patch.object(read_parameters, patched_method, return_value=return_value) as mock_plot:
        assert (
            read_parameters._generate_plots(
                parameter, parameter_version, input_file, tmp_test_directory
            )
            == return_value
        )
        if patched_method == "_plot_parameter_tables":
            mock_plot.assert_called_once_with(parameter, parameter_version, tmp_test_directory)
        else:
            mock_plot.assert_called_once_with(
                parameter, parameter_version, input_file, tmp_test_directory
            )


def test__plot_parameter_tables(tmp_test_directory, mocker):
    from pathlib import Path

    from simtools.visualization import plot_tables

    args = {"telescope": "LSTN-design", "site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(args=args, output_path=tmp_test_directory)

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
    mocker.patch.object(plot_tables, "plot")

    result = read_parameters._plot_parameter_tables(
        "pm_photoelectron_spectrum", "1.0.0", Path(tmp_test_directory)
    )
    assert result == ["pm_photoelectron_spectrum_1.0.0_North_LSTN-design"]

    args = {"telescope": None, "site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(args=args, output_path=tmp_test_directory)

    # Mock to return None (empty)
    mocker.patch.object(plot_tables, "generate_plot_configurations", return_value=None)

    result = read_parameters._plot_parameter_tables(
        "camera_config_file", "1.0.0", Path(tmp_test_directory)
    )
    assert result == []


def test__format_parameter_value(tmp_path):
    read_parameters = ReadParameters(args={}, output_path=tmp_path)
    parameter_name = "test"

    mock_data_1 = [[24.74, 9.0, 350.0, 1066.0], ["ns", "ns", "V", "V"], False, "1.0.0"]
    result_1 = read_parameters._format_parameter_value(parameter_name, *mock_data_1)
    assert result_1 == "24.74 ns, 9.0 ns, 350.0 V, 1066.0 V"

    mock_data_2 = [4.0, " ", False, "1.0.0"]
    result_2 = read_parameters._format_parameter_value(parameter_name, *mock_data_2)
    assert result_2 == "4.0"

    mock_data_3 = [[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], "GHz", False, "1.0.0"]
    result_3 = read_parameters._format_parameter_value(parameter_name, *mock_data_3)
    assert result_3 == "all: 0.2 GHz"

    mock_data_4 = [[1, 2, 3, 4], "m", False, "1.0.0"]
    result_4 = read_parameters._format_parameter_value(parameter_name, *mock_data_4)
    assert result_4 == "1 m, 2 m, 3 m, 4 m"

    mock_data_5 = [{"a": 1, "b": 2}, "m", False, "1.0.0"]
    result_5 = read_parameters._format_parameter_value(parameter_name, *mock_data_5)
    assert result_5 == "{'a': 1, 'b': 2} m"

    mock_data_6 = [[{"a": 1}, {"b": 2}, {"c": 3}], "m", False, "1.0.0"]
    result_6 = read_parameters._format_parameter_value(parameter_name, *mock_data_6)
    assert result_6 == "[View Test](#test)"


def test__group_model_versions_by_parameter_version(tmp_path):
    read_parameters = ReadParameters(args={}, output_path=tmp_path)

    mock_data = {
        "nsb_pixel_rate": [
            {
                "value": "all: 0.233591 GHz",
                "parameter_version": "2.0.0",
                "model_version": "6.0.0",
                "file_flag": False,
            },
            {
                "value": "all: 0.238006 GHz",
                "parameter_version": "1.0.0",
                "model_version": "5.0.0",
                "file_flag": False,
            },
        ],
        "pm_gain_index": [
            {
                "value": "4.5",
                "parameter_version": "1.0.0",
                "model_version": "6.0.0",
                "file_flag": False,
            },
            {
                "value": "4.5",
                "parameter_version": "1.0.0",
                "model_version": "5.0.0",
                "file_flag": False,
            },
        ],
    }

    expected = {
        "nsb_pixel_rate": [
            {
                "value": "all: 0.233591 GHz",
                "parameter_version": "2.0.0",
                "file_flag": False,
                "dict_table": None,
                "dict_unit": None,
                "model_version": "6.0.0",
            },
            {
                "value": "all: 0.238006 GHz",
                "parameter_version": "1.0.0",
                "file_flag": False,
                "dict_table": None,
                "dict_unit": None,
                "model_version": "5.0.0",
            },
        ],
        "pm_gain_index": [
            {
                "value": "4.5",
                "parameter_version": "1.0.0",
                "file_flag": False,
                "dict_table": None,
                "dict_unit": None,
                "model_version": "5.0.0, 6.0.0",
            }
        ],
    }

    result = read_parameters._group_model_versions_by_parameter_version(mock_data)

    assert result == expected


def test__compare_parameter_across_versions(tmp_test_directory, mocker):
    args = {"site": "North", "telescope": "LSTN-01"}
    output_path = tmp_test_directory
    read_parameters = ReadParameters(args=args, output_path=output_path)

    # Mock get_model_versions to return versions matching mock_data
    mocker.patch.object(read_parameters.db, "get_model_versions", return_value=["5.0.0", "6.0.0"])

    mock_data = {
        "5.0.0": {
            "quantum_efficiency": {
                "instrument": "LSTN-01",
                "site": "North",
                "parameter_version": "1.0.0",
                "value": QE_FILE_NAME,
                "unit": None,
                "file": True,
            },
            "array_element_position_ground": {
                "instrument": "LSTN-01",
                "site": "North",
                "parameter_version": "1.0.0",
                "value": [-70.93, -52.07, 43.0],
                "unit": "m",
                "file": False,
            },
        },
        "6.0.0": {
            "quantum_efficiency": {
                "instrument": "LSTN-01",
                "site": "North",
                "parameter_version": "1.0.0",
                "value": QE_FILE_NAME,
                "unit": None,
                "file": True,
            },
            "array_element_position_ground": {
                "instrument": "LSTN-01",
                "site": "North",
                "parameter_version": "2.0.0",
                "value": [-70.91, -52.35, 45.0],
                "unit": "m",
                "file": False,
            },
            "only_prod6_param": {
                "instrument": "LSTN-01",
                "site": "North",
                "parameter_version": "1.0.0",
                "value": 70.45,
                "unit": "m",
                "file": False,
            },
            "none_valued_param": {
                "instrument": "LSTN-01",
                "site": "North",
                "parameter_version": "1.0.0",
                "value": None,
                "unit": None,
                "file": False,
            },
        },
    }

    comparison_data = read_parameters._compare_parameter_across_versions(
        mock_data,
        [
            "quantum_efficiency",
            "array_element_position_ground",
            "only_prod6_param",
            "none_valued_param",
        ],
    )

    # Find quantum_efficiency comparison entry for parameter_version == "1.0.0"
    qe_comparison = comparison_data.get("quantum_efficiency")
    qe_versions = [
        entry["model_version"] for entry in qe_comparison if entry["parameter_version"] == "1.0.0"
    ]
    # Should be a single entry with model_version "5.0.0, 6.0.0"
    assert any("5.0.0" in v and "6.0.0" in v for v in qe_versions)

    position_comparison = comparison_data.get("array_element_position_ground")
    # Should have two entries with different model_version values
    assert len(position_comparison) == 2
    assert position_comparison[0]["model_version"] != position_comparison[1]["model_version"]
    # Find entry for parameter_version == "2.0.0"
    pos_versions = [
        entry["model_version"]
        for entry in position_comparison
        if entry["parameter_version"] == "2.0.0"
    ]
    assert pos_versions == ["6.0.0"]

    only_prod6_param = comparison_data.get("only_prod6_param")
    assert len(only_prod6_param) == 1
    assert only_prod6_param[0]["model_version"] == "6.0.0"

    assert "none_valued_param" not in comparison_data


def test_produce_observatory_report(mocker, tmp_path):
    rp = ReadParameters({"site": "North", "model_version": "6.0.0"}, tmp_path)

    # Empty data -> warning, no file
    mock_logger = mocker.patch("logging.Logger.warning")
    with patch.object(rp.db, "get_model_parameters", return_value={}):
        rp.produce_observatory_report()
    mock_logger.assert_called_once()
    assert not (tmp_path / "OBS-North.md").exists()

    # Full data -> file with layouts, triggers, skips None-valued param
    mock_params = {
        "site_elevation": {"value": 2200, "unit": "m", "parameter_version": "1.0"},
        "array_layouts": {
            "value": [{"name": "L1", "elements": ["LST1"]}],
            "unit": None,
            "parameter_version": "1.0",
        },
        "array_triggers": {
            "value": [
                {
                    "name": "T1",
                    "multiplicity": {"value": 2, "unit": None},
                    "width": {"value": 120, "unit": "ns"},
                    "hard_stereo": {"value": True, "unit": None},
                    "min_separation": {"value": None, "unit": "m"},
                }
            ],
            "unit": None,
            "parameter_version": "1.0",
        },
        "none_valued_param": {"value": None, "unit": None, "parameter_version": "1.0"},
    }
    with patch.object(rp.db, "get_model_parameters", return_value=mock_params):
        rp.produce_observatory_report()
    content = (tmp_path / "OBS-North.md").read_text()
    assert "# Observatory Parameters" in content
    assert "| site_elevation | 2200 m |" in content
    assert "none_valued_param" not in content
    assert "## Array Layouts" in content
    assert "## Array Trigger Configurations" in content


def test__write_parameters_table(tmp_path):
    """Test writing parameters table."""
    args = {}
    read_parameters = ReadParameters(args, tmp_path)

    mock_params = {
        "site_elevation": {"value": 2200, "unit": "m", "parameter_version": "1.0.0"},
        "array_layouts": {"value": [], "unit": None, "parameter_version": "2.0.0"},
        "array_triggers": {"value": [], "unit": None, "parameter_version": "3.0.0"},
        "dict_param": {
            "value": [{"a": "A", "b": "B", "value": 1}, {"a": "X", "b": "Y", "value": 2}],
            "unit": "m",
            "parameter_version": "4.0.0",
        },
    }

    with StringIO() as file:
        read_parameters._write_parameters_table(file, mock_params)
        output = file.getvalue()

    # Verify table headers
    assert "| Parameter | Value | Parameter Version |" in output

    # Verify normal parameter
    assert "| site_elevation | 2200 m | 1.0.0 |" in output

    # Verify special sections
    assert "| array_layouts | [View Array Layouts](#array-layouts) | 2.0.0 |" in output
    assert (
        "| array_triggers | [View Trigger Configurations](#array-trigger-configurations) | 3.0.0 |"
        in output
    )

    # Verify list-of-dicts parameter is linked and table is written below
    assert "| dict_param | [View Dict Param](#dict-param) | 4.0.0 |" in output
    assert "## Dict Param" in output
    assert "| a | b | value |" in output
    assert "| A | B | 1 m |" in output
    assert "| X | Y | 2 m |" in output


def test_model_version_setter_with_invalid_list(tmp_path):
    """Test setting model_version with an invalid list containing more than one element."""
    args = {"model_version": "6.0.0"}
    read_parameters = ReadParameters(args=args, output_path=tmp_path)

    error_message = "Only one model version can be passed to ReadParameters, not a list."

    with pytest.raises(ValueError, match=error_message):
        read_parameters.model_version = ["7.0.0"]

    with pytest.raises(ValueError, match=error_message):
        read_parameters.model_version = ["7.0.0", "8.0.0"]

    with pytest.raises(ValueError, match=error_message):
        read_parameters.model_version = []


_SIM_PARAM = {
    "p": {"value": 5.0, "unit": "", "parameter_version": "1.0.0"},
    "none_p": {"value": None, "unit": "", "parameter_version": "1.0.0"},
}
_SIM_DESC = {
    "p": {"description": "D", "short_description": "S"},
    "none_p": {"description": "D", "short_description": None},
}


@pytest.mark.parametrize("simulation_software", ["corsika", "sim_telarray"])
def test_get_simulation_configuration_data(simulation_software, tmp_path):
    rp = ReadParameters(
        args={
            "model_version": "6.0.0",
            "simulation_software": simulation_software,
            "telescope": "LSTN-01",
            "site": "North",
        },
        output_path=tmp_path,
    )
    with (
        patch.object(rp, "get_all_parameter_descriptions", return_value=_SIM_DESC),
        patch.object(rp.db, "export_model_files"),
        patch.object(rp.db, "get_simulation_configuration_parameters", return_value=_SIM_PARAM),
        patch.object(rp.db, "get_array_elements", return_value=["LSTN-01"]),
        patch("simtools.utils.names.get_site_from_array_element_name", return_value="North"),
    ):
        data, dict_tables = rp.get_simulation_configuration_data()
    assert len(data) == 1
    assert data[0][3] == "5.0"
    assert dict_tables == []


def test__write_to_file(telescope_model_lst, tmp_path):
    args = {
        "telescope": telescope_model_lst.name,
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
        "simulation_software": "corsika",
    }
    read_parameters = ReadParameters(args=args, output_path=tmp_path)

    mock_data = [
        [telescope_model_lst.name, "param_1", "1.0.0", "5.0", "Full description 1", "Short 1"],
        [
            telescope_model_lst.name,
            "param_2",
            "2.1.0",
            "0.3 GeV, 0.2 GeV",
            "Energy cutoff details",
            "Short 2",
        ],
    ]

    output_file = tmp_path / "output.md"

    with output_file.open("w") as f:
        read_parameters._write_to_file(mock_data, f)

    result = output_file.read_text()

    assert "| Parameter Name" in result
    assert "| param_1" in result
    assert "Short 1" in result
    assert "0.3 GeV, 0.2 GeV" in result


def test_produce_simulation_configuration_report(tmp_path):
    args = {
        "telescope": "LSTN-01",
        "site": "North",
        "model_version": "6.0.0",
        "simulation_software": "sim_telarray",
    }
    read_parameters = ReadParameters(args=args, output_path=tmp_path)

    mock_data = [
        (
            "LSTN-01",
            "iobuf maximum",
            "1.0.0",
            "100 byte",
            "Description",
            "Buffer limits for input and output of eventio data.",
        ),
        (
            "LSTN-01",
            "random generator",
            "1.0.0",
            "mt19937",
            "Random generator used.",
            None,
        ),
    ]

    with patch.object(
        read_parameters,
        "get_simulation_configuration_data",
        return_value=(mock_data, []),
    ):
        read_parameters.produce_simulation_configuration_report()

        report_file = tmp_path / f"configuration_{read_parameters.software}.md"
        assert report_file.exists()

        content = report_file.read_text()

        assert f"# configuration_{read_parameters.software}" in content
        assert "| Parameter Name" in content
        assert "Buffer limits for input and output of eventio data." in content
        assert "mt19937" in content

    # testing for corsika
    args["simulation_software"] = "corsika"
    read_parameters_corsika = ReadParameters(args=args, output_path=tmp_path)

    mock_data_corsika = [
        (
            "LSTN-01",
            "corsika_starting_grammage",
            "1.0.0",
            "[View Corsika Starting Grammage](#corsika-starting-grammage)",
            "Description",
            "Short",
        )
    ]
    mock_dict_tables = [
        (
            "LSTN-01",
            "corsika_starting_grammage",
            [{"instrument": "LSTN-design", "primary_particle": "muon+", "value": 580.0}],
            "g/cm2",
        )
    ]

    with patch.object(
        read_parameters_corsika,
        "get_simulation_configuration_data",
        return_value=(mock_data_corsika, mock_dict_tables),
    ):
        read_parameters_corsika.produce_simulation_configuration_report()

        report_file_corsika = tmp_path / f"configuration_{read_parameters_corsika.software}.md"
        assert report_file_corsika.exists()

        content = report_file_corsika.read_text()
        assert "# configuration_corsika" in content
        assert "[View Corsika Starting Grammage](#corsika-starting-grammage)" in content
        assert "## Corsika Starting Grammage" in content
        assert "| instrument | primary_particle | value |" in content
        assert "| LSTN-design | muon+ | 580.0 g/cm2 |" in content


def test_produce_calibration_reports(mocker, tmp_path):
    rp = ReadParameters(args={"model_version": "6.0.0"}, output_path=tmp_path)
    mocker.patch.object(rp.db, "get_array_elements", return_value=["ILLN-01"])
    mocker.patch.object(rp.db, "get_design_model", return_value="ILLN-design")
    mocker.patch.object(
        rp.db,
        "get_model_parameters",
        return_value={
            "laser_events": {
                "value": 10,
                "unit": None,
                "parameter_version": "1.0.0",
                "instrument": "ILLN-design",
            },
            "array_element_position_ground": {
                "value": [0.0, 0.0, 0.0],
                "unit": "m",
                "parameter_version": "1.0.0",
                "instrument": "ILLN-01",
            },
        },
    )
    desc = {
        "laser_events": {
            "description": "Laser",
            "short_description": None,
            "inst_class": "Calibration",
        },
        "array_element_position_ground": {
            "description": "Pos",
            "short_description": "P",
            "inst_class": "Structure",
        },
    }
    with patch.object(rp, "get_all_parameter_descriptions", return_value=desc):
        rp.produce_calibration_reports()
    content = (tmp_path / "ILLN-01.md").read_text()
    assert "# ILLN-01" in content
    assert "laser events" in content


def test_get_calibration_data(tmp_path):
    rp = ReadParameters(args={"model_version": "6.0.0"}, output_path=tmp_path)
    mock_data = {
        "dark_events": {
            "value": 1,
            "unit": None,
            "parameter_version": "1.0.0",
            "instrument": "ILLN-design",
        },
        "laser_events": {
            "value": 10,
            "unit": None,
            "parameter_version": "1.0.0",
            "instrument": "ILLN-01",
        },
        "none_value": {
            "value": None,
            "unit": None,
            "parameter_version": "1.0.0",
            "instrument": "ILLN-design",
        },
        "dict_parameter": {
            "value": [{"channel": 1, "value": 12.0}],
            "unit": "p.e.",
            "parameter_version": "1.0.0",
            "instrument": "ILLN-design",
        },
        "array_element_position_ground": {
            "value": [0.0, 0.0, 0.0],
            "unit": "m",
            "parameter_version": "1.0.0",
            "file": False,
            "instrument": "ILLN-design",
        },
    }
    calib_desc = {
        "dark_events": {
            "description": "Dark",
            "short_description": "D",
            "inst_class": "Calibration",
        },
        "laser_events": {
            "description": "Laser",
            "short_description": None,
            "inst_class": "Calibration",
        },
        "none_value": {
            "description": "None",
            "short_description": "N",
            "inst_class": "Calibration",
        },
        "dict_parameter": {
            "description": "Dict",
            "short_description": "D",
            "inst_class": "Calibration",
        },
    }
    tel_desc = {
        "array_element_position_ground": {
            "description": "pos desc",
            "short_description": "pos",
            "inst_class": "Structure",
        }
    }
    with patch.object(
        rp,
        "get_all_parameter_descriptions",
        side_effect=[calib_desc, tel_desc, calib_desc, tel_desc],
    ):
        result, dict_tables = rp.get_calibration_data(mock_data, "ILLN-01")
    assert result[0][0] == "Structure"  # reverse-sorted: Structure before Calibration
    assert len(result) == 4
    assert dict_tables == [("dict_parameter", [{"channel": 1, "value": 12.0}], "p.e.")]
    structure_row = next(r for r in result if r[0] == "Structure")
    assert "array_element_position_ground" in structure_row[1]


class DummyTelescope:
    def __init__(self, site, name, model_version, param_versions=None, design_model=None):
        self.site = site
        self.name = name
        self.model_version = model_version
        self._param_versions = param_versions or {}
        self.design_model = design_model or name

    def get_parameter_version(self, parameter):
        return self._param_versions.get(parameter)


def test_get_array_element_parameter_data_simple(tmp_test_directory, monkeypatch):
    rp = ReadParameters(
        args={"telescope": "LSTN-01", "site": "North", "model_version": "1.0.0"},
        output_path=tmp_test_directory,
    )
    rp.db = Mock()
    rp.db.export_model_files.return_value = None
    rp.get_all_parameter_descriptions = Mock(
        return_value={
            "test_param": {
                "description": DESCRIPTION,
                "short_description": SHORT_DESC,
                "inst_class": "Telescope",
            },
            "none_param": {
                "description": "none",
                "short_description": "n",
                "inst_class": "Telescope",
            },
        }
    )
    monkeypatch.setattr(names, "is_design_type", lambda _name: False)

    # None-valued param is skipped; OTHER instrument is not emphasised
    rp.db.get_model_parameters.return_value = {
        "test_param": {
            "unit": "m",
            "value": 42,
            "parameter_version": "1.0.0",
            "file": False,
            "instrument": "OTHER",
        },
        "none_param": {
            "unit": "m",
            "value": None,
            "parameter_version": "1.0.0",
            "file": False,
            "instrument": "OTHER",
        },
    }
    monkeypatch.setattr(
        names, "model_parameters", lambda *a, **k: {"test_param": {}, "none_param": {}}
    )
    tel = DummyTelescope(
        "North", "LSTN-01", "1.0.0", {"test_param": "1.0.0", "none_param": "1.0.0"}
    )
    data, _dt = rp.get_array_element_parameter_data(telescope_model=tel)
    assert data == [["Telescope", "test_param", "1.0.0", "42 m", DESCRIPTION, SHORT_DESC]]

    # Instrument match -> bold emphasis
    rp.db.get_model_parameters.return_value = {
        "test_param": {
            "unit": "m",
            "value": 42,
            "parameter_version": "1.0.0",
            "file": False,
            "instrument": "LSTN-01",
        },
    }
    monkeypatch.setattr(names, "model_parameters", lambda *a, **k: {"test_param": {}})
    data, _dt = rp.get_array_element_parameter_data(telescope_model=tel)
    assert data[0][1] == "***test_param***"


def test_get_array_element_parameter_data_file_parameter(tmp_test_directory, monkeypatch):
    rp = ReadParameters(
        args={"telescope": "LSTN-01", "site": "North", "model_version": "1.0.0"},
        output_path=tmp_test_directory,
    )
    rp.db = Mock()
    rp.db.get_model_parameters.return_value = {
        "file_param": {
            "unit": None,
            "value": "myfile.dat",
            "parameter_version": "1.0.0",
            "file": True,
            "instrument": "OTHER",
        }
    }
    rp.db.export_model_files.return_value = None
    rp.get_all_parameter_descriptions = Mock(
        return_value={
            "file_param": {
                "description": DESCRIPTION,
                "short_description": SHORT_DESC,
                "inst_class": "Telescope",
            }
        }
    )
    monkeypatch.setattr(
        ReadParameters, "_convert_to_md", lambda self, p, pv, f: "_data_files/myfile.md"
    )
    monkeypatch.setattr(names, "model_parameters", lambda *a, **k: {"file_param": {}})
    monkeypatch.setattr(names, "is_design_type", lambda _name: False)

    tel = DummyTelescope("North", "LSTN-01", "1.0.0", {"file_param": "1.0.0"})
    data, _dt = rp.get_array_element_parameter_data(telescope_model=tel)
    assert data[0][3] == "[myfile.dat](_data_files/myfile.md)"


@pytest.mark.parametrize(
    ("parameter_version", "preexisting_plot"),
    [(None, False), ("1.0.0", False), ("1.0.0", True)],
)
def test__plot_camera_config(tmp_test_directory, mocker, parameter_version, preexisting_plot):
    read_parameters = ReadParameters(
        args={"telescope": "LSTN-01", "site": "North", "model_version": "6.0.0"},
        output_path=tmp_test_directory,
    )
    input_file = Path(tmp_test_directory / "camera_config_file.dat")
    input_file.touch()
    plot_name = input_file.stem.replace(".", "-")
    plot_path = Path(tmp_test_directory / f"{plot_name}.png")
    if preexisting_plot:
        plot_path.touch()

    mock_plot = mocker.patch("simtools.visualization.plot_pixels.plot")
    result = read_parameters._plot_camera_config(
        "camera_config_file", parameter_version, input_file, tmp_test_directory
    )

    assert result == ([] if parameter_version is None else [plot_name])
    if parameter_version is None or preexisting_plot:
        mock_plot.assert_not_called()
    else:
        mock_plot.assert_called_once_with(
            config={
                "file_name": input_file.name,
                "telescope": "LSTN-01",
                "parameter_version": "1.0.0",
                "site": "North",
                "model_version": "6.0.0",
                "parameter": "camera_config_file",
            },
            output_file=plot_path.with_suffix(""),
        )


def test_is_markdown_link():
    read_parameters = ReadParameters(args={}, output_path=Path())

    assert read_parameters.is_markdown_link("[example](http://example.com)") is True
    assert read_parameters.is_markdown_link("[text](target)") is True
    assert read_parameters.is_markdown_link("not a link") is False
    assert read_parameters.is_markdown_link("[missing target]") is False
    assert read_parameters.is_markdown_link("(missing text)") is False
    assert read_parameters.is_markdown_link("[text](target") is False
    assert read_parameters.is_markdown_link("[text]target)") is False
    assert read_parameters.is_markdown_link("") is False


@pytest.mark.parametrize(
    ("parameter", "latest_value", "expected_plot_text"),
    [
        ("param_x", None, "![Parameter plot.]"),
        ("camera_config_file", "[cam_config.dat](link)", "![Camera configuration plot.]"),
        ("camera_config_file", None, None),
        ("camera_config_file", "no link here", None),
    ],
)
def test_write_file_flag_section(parameter, latest_value, expected_plot_text, tmp_path):
    rp = ReadParameters(
        args={"telescope": "LSTN-01", "site": "North", "model_version": "6.0.0"},
        output_path=tmp_path,
    )
    rp._get_telescope_identifier = lambda mv=None: "LSTN-01"

    comparison_data = {
        parameter: [
            {
                "parameter_version": "1.0.0",
                "model_version": "6.0.0",
                "file_flag": True,
                **({} if parameter == "param_x" else {"value": latest_value}),
            }
        ]
    }

    buf = StringIO()
    with patch("simtools.io.io_handler.IOHandler.get_output_directory", return_value=tmp_path):
        rp._write_file_flag_section(buf, parameter, comparison_data)
    content = buf.getvalue()

    assert "The latest parameter version is plotted below." in content
    if expected_plot_text is None:
        assert "![Camera configuration plot.]" not in content
    else:
        assert expected_plot_text in content


@pytest.mark.parametrize(
    ("calibration_device", "data", "dict_tables", "expected_strings"),
    [
        (
            "ILLN-01",
            [
                ["Calibration", "***laser_events***", "1.0.0", "10", "Desc", None],
                ["Calibration", "pedestal_events", "1.0.0", "100", "Desc2", "Short2"],
            ],
            None,
            ["***laser events***", "pedestal events"],
        ),
        (
            "ILLN-02",
            [["Calibration", 12345, "1.0.0", "10", "Desc", None]],
            [("dict_parameter", [{"a": 1, "value": 2}], "m")],
            ["12345"],
        ),
    ],
)
def test_write_single_calibration_report(
    tmp_path, calibration_device, data, dict_tables, expected_strings
):
    output_file = tmp_path / f"{calibration_device}.md"
    rp = ReadParameters(args={"model_version": "6.0.0"}, output_path=tmp_path)

    with (
        patch.object(names, "is_design_type", return_value=False),
        patch.object(rp, "_write_dict_table") as write_dict_table,
    ):
        rp._write_single_calibration_report(
            output_file,
            calibration_device,
            data,
            "ILLN-design",
            dict_tables=dict_tables,
        )

    assert write_dict_table.call_count == int(dict_tables is not None)
    if dict_tables is not None:
        assert write_dict_table.call_args[0][0] == "dict_parameter"
        assert write_dict_table.call_args[0][2] == [{"a": 1, "value": 2}]

    content = output_file.read_text(encoding="utf-8")
    for expected in expected_strings:
        assert expected in content


def test_generate_model_parameter_reports_for_devices(tmp_path):
    """Test generating model parameter reports for calibration devices."""
    args = {"all_sites": True, "all_telescopes": True}
    read_parameters = ReadParameters(args=args, output_path=tmp_path)

    array_elements = ["ILLN-01", "ILLN-02", "ILLS-01"]

    # Mock the site resolution and produce_model_parameter_reports
    with patch("simtools.utils.names.get_site_from_array_element_name") as mock_get_site:
        mock_get_site.side_effect = lambda x: "North" if "ILLN" in x else "South"

        with patch.object(read_parameters, "produce_model_parameter_reports") as mock_produce:
            read_parameters.generate_model_parameter_reports_for_devices(array_elements)

            # Verify that produce_model_parameter_reports was called for each device
            assert mock_produce.call_count == 3

            # Verify that site and array_element were set correctly for each device
            expected_calls = [call(collection="calibration_devices")] * 3
            mock_produce.assert_has_calls(expected_calls)

            # Check that the last processed device set the correct attributes
            assert read_parameters.array_element == "ILLS-01"
            assert read_parameters.site == "South"


@pytest.mark.parametrize(
    ("value1", "value2", "expected"),
    [
        (None, 1, False),
        ([1, {"a": 2}], [1, {"a": 2}], True),
        ({"a": 1}, {"a": 1, "b": 2}, False),
        ("same", "same", True),
    ],
)
def test__values_equal_branches(value1, value2, expected, tmp_path):
    rp = ReadParameters(args={}, output_path=tmp_path)
    assert rp._values_equal(value1, value2) is expected


def test__parameter_changed_branches(tmp_path):
    rp = ReadParameters(args={}, output_path=tmp_path)
    base = {"parameter_version": "1.0.0", "unit": "m", "file": False, "value": 1}

    assert rp._parameter_changed(base, None) is True
    assert rp._parameter_changed(base, {**base, "parameter_version": "1.0.1"}) is True
    assert rp._parameter_changed(base, {**base, "unit": "cm"}) is True
    assert rp._parameter_changed(base, {**base, "file": True}) is True
    assert rp._parameter_changed(base, {**base, "value": 2}) is True


def test__convert_to_md_writes_plot_references(tmp_path, mocker):
    rp = ReadParameters(args={}, output_path=tmp_path)
    input_file = tmp_path / "parameter_data.dat"
    input_file.write_text("\n".join(f"line {i}" for i in range(40)), encoding="utf-8")

    mocker.patch.object(rp, "_generate_plots", return_value=["plot_a", "plot_b"])
    relative_md = rp._convert_to_md("param", "1.0.0", input_file)

    content = (tmp_path / relative_md).read_text(encoding="utf-8")
    assert "![Parameter plot.]" in content
    assert "plot_a.png" in content
    assert "plot_b.png" in content


def test__is_list_of_dicts_empty_list(tmp_path):
    rp = ReadParameters(args={}, output_path=tmp_path)
    assert rp._is_list_of_dicts([]) is False


def test_get_array_element_parameter_data_collects_dict_table(tmp_test_directory, monkeypatch):
    rp = ReadParameters(
        args={"telescope": "LSTN-01", "site": "North", "model_version": "1.0.0"},
        output_path=tmp_test_directory,
    )
    rp.db = Mock()
    rp.db.get_model_parameters.return_value = {
        "dict_param": {
            "unit": "m",
            "value": [{"pixel": 1, "value": 0.1}],
            "parameter_version": "1.0.0",
            "file": False,
            "instrument": "OTHER",
        }
    }
    rp.db.export_model_files.return_value = None
    rp.get_all_parameter_descriptions = Mock(
        return_value={
            "dict_param": {
                "description": "desc",
                "short_description": "short",
                "inst_class": "Camera",
            }
        }
    )

    monkeypatch.setattr(names, "model_parameters", lambda *args, **kwargs: {"dict_param": {}})
    monkeypatch.setattr(names, "is_design_type", lambda _name: False)

    tel = DummyTelescope("North", "LSTN-01", "1.0.0", {"dict_param": "1.0.0"})
    _data, dict_tables = rp.get_array_element_parameter_data(telescope_model=tel)
    assert len(dict_tables) == 1


def test_get_simulation_configuration_data_collects_dict_table(tmp_path):
    rp = ReadParameters(
        args={
            "model_version": "6.0.0",
            "simulation_software": "corsika",
            "telescope": "LSTN-01",
            "site": "North",
        },
        output_path=tmp_path,
    )

    with (
        patch.object(
            rp.db,
            "get_simulation_configuration_parameters",
            return_value={
                "dict_param": {
                    "value": [{"particle": "gamma", "value": 1.0}],
                    "unit": "GeV",
                    "parameter_version": "1.0.0",
                    "file": False,
                }
            },
        ),
        patch.object(
            rp,
            "get_all_parameter_descriptions",
            return_value={
                "dict_param": {
                    "description": "desc",
                    "short_description": "short",
                }
            },
        ),
        patch.object(rp.db, "export_model_files"),
    ):
        _data, dict_tables = rp.get_simulation_configuration_data()

    assert len(dict_tables) == 1
    assert dict_tables[0][1] == "dict_param"


def test_produce_simulation_configuration_report_non_simtel_writes_dict_table(tmp_path, mocker):
    rp = ReadParameters(
        args={"model_version": "6.0.0", "simulation_software": "corsika"},
        output_path=tmp_path,
    )
    mocker.patch.object(
        rp,
        "get_simulation_configuration_data",
        return_value=(
            [("LSTN-01", "p", "1.0.0", "v", "d", "s")],
            [("LSTN-01", "dict_param", [{"a": 1}], "m")],
        ),
    )
    write_dict_table = mocker.patch.object(rp, "_write_dict_table")

    rp.produce_simulation_configuration_report()

    assert write_dict_table.call_count == 1
    assert write_dict_table.call_args[0][0] == "dict_param"


def test_produce_array_element_report_writes_design_note_and_dict_table(tmp_path, mocker):
    rp = ReadParameters(
        args={"site": "North", "telescope": "LSTN-01", "model_version": "6.0.0"},
        output_path=tmp_path,
    )
    mocker.patch(
        "simtools.reporting.docs_read_parameters.TelescopeModel",
        return_value=DummyTelescope(
            site="North",
            name="LSTN-01",
            model_version="6.0.0",
            design_model="LSTN-design",
        ),
    )

    mocker.patch.object(
        rp,
        "get_array_element_parameter_data",
        return_value=(
            [["Structure", "param", "1.0.0", "1 m", "desc", "short"]],
            [("dict_param", [{"a": 1, "value": 2}], "m")],
        ),
    )
    write_dict_table = mocker.patch.object(rp, "_write_dict_table")

    rp.produce_array_element_report()

    content = (tmp_path / "LSTN-01.md").read_text(encoding="utf-8")
    assert "The design model can be found here" in content
    assert "Parameters shown in ***bold and italics***" in content
    assert write_dict_table.call_count == 1


def test_produce_model_parameter_reports_writes_latest_dict_table(tmp_path, mocker):
    rp = ReadParameters(
        args={"site": "North", "telescope": "LSTN-01", "model_version": "6.0.0"},
        output_path=tmp_path,
    )

    mocker.patch("simtools.utils.names.model_parameters", return_value={"dict_param": {}})
    mocker.patch.object(rp.db, "get_model_parameters_for_all_model_versions", return_value={})
    mocker.patch.object(
        rp,
        "_compare_parameter_across_versions",
        return_value={
            "dict_param": [
                {
                    "parameter_version": "1.0.0",
                    "model_version": "5.0.0",
                    "value": "v1",
                    "file_flag": False,
                    "dict_table": [{"a": 1, "value": 1}],
                    "dict_unit": "m",
                },
                {
                    "parameter_version": "2.0.0",
                    "model_version": "6.0.0",
                    "value": "v2",
                    "file_flag": False,
                    "dict_table": [{"a": 2, "value": 2}],
                    "dict_unit": "m",
                },
            ]
        },
    )
    mocker.patch.object(
        rp,
        "get_all_parameter_descriptions",
        return_value={"dict_param": {"description": "desc"}},
    )
    write_dict_table = mocker.patch.object(rp, "_write_dict_table")

    rp.produce_model_parameter_reports()

    assert write_dict_table.call_count == 1
    assert write_dict_table.call_args[0][0] == "dict_param"
    assert write_dict_table.call_args[0][2] == [{"a": 2, "value": 2}]


def test__write_dict_table_empty_value_data(tmp_path):
    rp = ReadParameters(args={}, output_path=tmp_path)
    file = StringIO()
    rp._write_dict_table("dict_param", file, [], "m")
    assert file.getvalue() == ""


def test_get_calibration_data_uses_telescope_description_fallback(tmp_path):
    rp = ReadParameters(args={"model_version": "6.0.0"}, output_path=tmp_path)

    all_parameter_data = {
        "array_element_position_ground": {
            "value": [0.0, 0.0, 0.0],
            "unit": "m",
            "parameter_version": "1.0.0",
            "file": False,
            "instrument": "ILLN-design",
        }
    }

    with patch.object(
        rp,
        "get_all_parameter_descriptions",
        side_effect=[
            {},
            {
                "array_element_position_ground": {
                    "description": "pos desc",
                    "short_description": "pos short",
                    "inst_class": "Structure",
                }
            },
        ],
    ):
        data, _dict_tables = rp.get_calibration_data(all_parameter_data, "ILLN-01")

    assert data[0][0] == "Structure"
    assert "array_element_position_ground" in data[0][1]


def test_generate_model_parameter_reports_for_devices_site_list(tmp_path):
    rp = ReadParameters(args={"all_sites": True}, output_path=tmp_path)

    with (
        patch(
            "simtools.utils.names.get_site_from_array_element_name",
            return_value=["North", "South"],
        ),
        patch.object(rp, "produce_model_parameter_reports") as mock_produce,
    ):
        rp.generate_model_parameter_reports_for_devices(["ILLN-01"])

    assert mock_produce.call_count == 1
    assert rp.site == "North"
    assert rp.array_element == "ILLN-01"


def test__build_version_parameter_item_returns_none_for_other_instrument(tmp_path):
    rp = ReadParameters(args={"telescope": "LSTN-01"}, output_path=tmp_path)

    result = rp._build_version_parameter_item(
        "6.0.0",
        "focal_length",
        {"instrument": "MSTN-01", "value": 28.0, "parameter_version": "1.0.0"},
    )

    assert result is None


def test__collect_calibration_array_elements_adds_design_model(tmp_path, mocker):
    rp = ReadParameters(args={"model_version": "6.0.0"}, output_path=tmp_path)
    mocker.patch.object(rp.db, "get_array_elements", return_value=["ILLN-01"])
    mocker.patch.object(rp.db, "get_design_model", return_value="ILLN-design")

    assert rp._collect_calibration_array_elements() == ["ILLN-01", "ILLN-design"]


def _delta_entry(value):
    return {"parameter_version": "1.0", "unit": "m", "file": False, "value": value}


DELTA_PARAM_DATA = {"focal_length": _delta_entry(1)}
DELTA_SITE_DATA = {"site_elevation": _delta_entry(2200)}


@pytest.mark.parametrize(
    ("param_data", "contains"),
    [
        (None, "-"),
        ({}, "-"),
        ({"value": [{"a": 1}, {"b": 2}], "unit": None, "file": False}, "2 rows"),
        ({"value": "some_file.dat", "unit": " ", "file": True}, "some_file.dat"),
        ({"value": list(range(15)), "unit": "m", "file": False}, "..."),
        ({"value": 42.0, "unit": "m", "file": False}, "42.0"),
    ],
)
def test__format_value_for_delta(param_data, contains, tmp_path):
    rp = ReadParameters(args={}, output_path=tmp_path)
    assert contains in rp._format_value_for_delta("p", param_data)


def test__write_delta_helpers(tmp_path):
    rp = ReadParameters(args={"model_version": "6.0.1"}, output_path=tmp_path)
    buf = StringIO()
    rp._write_delta_report_header(buf, "My Title", "6.0.0", "../6.0.0/target.md")
    rp._write_delta_report_table(
        buf,
        [
            {
                "parameter": "focal_length",
                "base_param_version": "1.0.0",
                "new_param_version": "1.0.1",
                "base_value": "2800",
                "new_value": "2810",
            }
        ],
    )
    content = buf.getvalue()
    assert "# My Title" in content
    assert "6.0.1" in content
    assert "../6.0.0/target.md" in content
    assert "focal_length" in content
    assert "2810" in content


@pytest.mark.parametrize(
    ("parameters", "base_data", "current_data", "expected_count"),
    [
        (
            None,
            {"a": _delta_entry(1)},
            {"a": _delta_entry(2)},
            1,
        ),
        (
            None,
            {"a": _delta_entry(1)},
            {"a": _delta_entry(1)},
            0,
        ),
    ],
)
def test__build_delta_changes(parameters, base_data, current_data, expected_count, tmp_path):
    rp = ReadParameters(args={}, output_path=tmp_path)
    changes = rp._build_delta_changes(base_data, current_data, parameters=parameters)
    assert len(changes) == expected_count


@pytest.mark.parametrize(
    ("method_name", "args", "base_version", "base_data", "current_data"),
    [
        (
            "_produce_array_element_delta_report",
            {"site": "North", "telescope": "LSTN-01", "model_version": "6.0.1"},
            "6.0.0",
            {},
            None,
        ),
        (
            "_produce_array_element_delta_report",
            {"site": "North", "telescope": "LSTN-01", "model_version": "6.0.1"},
            "6.0.0",
            DELTA_PARAM_DATA,
            {},
        ),
        (
            "_produce_observatory_delta_report",
            {"site": "South", "model_version": "6.2.1"},
            "6.2.0",
            {},
            None,
        ),
        (
            "_produce_observatory_delta_report",
            {"site": "South", "model_version": "6.2.1"},
            "6.2.0",
            DELTA_SITE_DATA,
            {},
        ),
    ],
)
def test__delta_report_returns_false_when_base_or_current_missing(
    method_name, args, base_version, base_data, current_data, tmp_path, mocker
):
    rp = ReadParameters(args=args, output_path=tmp_path)
    get_model_parameters = mocker.patch.object(rp.db, "get_model_parameters")
    if current_data is None:
        get_model_parameters.return_value = base_data
    else:
        get_model_parameters.side_effect = [base_data, current_data]

    report_method = getattr(rp, method_name)
    assert report_method(base_version) is False


@pytest.mark.parametrize(
    (
        "method_name",
        "args",
        "base_version",
        "param_data",
        "output_filename",
        "patch_names_model_parameters",
    ),
    [
        (
            "_produce_array_element_delta_report",
            {"site": "North", "telescope": "LSTN-01", "model_version": "6.0.1"},
            "6.0.0",
            DELTA_PARAM_DATA,
            "LSTN-01.md",
            True,
        ),
        (
            "_produce_observatory_delta_report",
            {"site": "South", "model_version": "6.2.1"},
            "6.2.0",
            DELTA_SITE_DATA,
            "OBS-South.md",
            False,
        ),
    ],
)
def test__delta_report_no_changes(
    method_name,
    args,
    base_version,
    param_data,
    output_filename,
    patch_names_model_parameters,
    tmp_path,
    mocker,
):
    rp = ReadParameters(args=args, output_path=tmp_path)
    mocker.patch.object(rp.db, "get_model_parameters", side_effect=[param_data, param_data])
    if patch_names_model_parameters:
        mocker.patch("simtools.utils.names.model_parameters", return_value={"focal_length": {}})

    report_method = getattr(rp, method_name)
    assert report_method(base_version) is True
    content = (tmp_path / output_filename).read_text()
    assert "No parameter changes detected" in content


@pytest.mark.parametrize(
    ("parameter_version", "preexisting_plot"),
    [(None, False), ("1.0.0", False), ("1.0.0", True)],
)
def test__plot_mirror_config(tmp_test_directory, mocker, parameter_version, preexisting_plot):
    read_parameters = ReadParameters(
        args={"telescope": "LSTN-01", "site": "North", "model_version": "6.0.0"},
        output_path=tmp_test_directory,
    )
    input_file = Path(tmp_test_directory / "mirror_config.dat")
    input_file.write_text("dummy mirror content")
    plot_name = input_file.stem.replace(".", "-")
    plot_path = Path(tmp_test_directory / f"{plot_name}.png")
    if preexisting_plot:
        plot_path.touch()

    mock_plot = mocker.patch("simtools.visualization.plot_mirrors.plot")
    result = read_parameters._plot_mirror_config(
        "mirror_list", parameter_version, input_file, tmp_test_directory
    )

    assert result == ([] if parameter_version is None else [plot_name])
    if parameter_version is None or preexisting_plot:
        mock_plot.assert_not_called()
    else:
        mock_plot.assert_called_once_with(
            config={
                "parameter": "mirror_list",
                "telescope": "LSTN-01",
                "parameter_version": "1.0.0",
                "site": "North",
                "model_version": "6.0.0",
            },
            output_file=Path(f"{tmp_test_directory}/{plot_name}"),
        )
