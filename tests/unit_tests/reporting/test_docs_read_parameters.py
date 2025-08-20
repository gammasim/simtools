import re
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from simtools.reporting.docs_read_parameters import ReadParameters
from simtools.utils import names

# Test constants
QE_FILE_NAME = "qe_lst1_20200318_high+low.dat"
DESCRIPTION = "Test parameter"
SHORT_DESC = "Short"


def test_get_all_parameter_descriptions(telescope_model_lst, io_handler, db_config):
    args = {
        "telescope": telescope_model_lst.name,
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
    }
    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)
    # Call get_all_parameter_descriptions
    description_dict = read_parameters.get_all_parameter_descriptions()

    assert isinstance(description_dict.get("focal_length"), dict)
    assert isinstance(description_dict.get("focal_length").get("description"), str)
    assert isinstance(description_dict.get("focal_length").get("short_description"), str)
    assert isinstance(description_dict.get("focal_length").get("inst_class"), str)


def test_produce_array_element_report(telescope_model_lst, io_handler, db_config, mocker):
    """Test array element report generation with both observatory and telescope scenarios."""
    # Test observatory report path
    args = {
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
        "observatory": True,
    }
    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

    # Mock observatory parameters
    mock_obs_params = {
        "site_elevation": {
            "value": 2200,
            "unit": "m",
        }
    }

    with patch.object(read_parameters.db, "get_model_parameters", return_value=mock_obs_params):
        read_parameters.produce_array_element_report()

        # Verify observatory report was generated
        obs_file = output_path / f"OBS-{args['site']}.md"
        assert obs_file.exists()

        # Verify DB was called with correct parameters
        read_parameters.db.get_model_parameters.assert_called_once_with(
            site=args["site"],
            array_element_name=f"OBS-{args['site']}",
            collection="sites",
            model_version=args["model_version"],
        )

    # Test telescope report
    args["observatory"] = False
    args["telescope"] = telescope_model_lst.name
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

    mock_telescope_params = {
        "focal_length": {
            "value": 2800.0,
            "unit": "cm",
            "parameter_version": "1.0.0",
            "instrument": telescope_model_lst.name,
        }
    }

    with patch.object(
        read_parameters.db, "get_model_parameters", return_value=mock_telescope_params
    ):
        read_parameters.produce_array_element_report()

        # Verify telescope report was generated
        tel_file = output_path / f"{telescope_model_lst.name}.md"
        assert tel_file.exists()

        # Verify DB was called with correct parameters
        read_parameters.db.get_model_parameters.assert_called_once_with(
            site=args["site"],
            array_element_name=args["telescope"],
            collection="telescopes",
            model_version=args["model_version"],
        )


def test_produce_model_parameter_reports(io_handler, db_config):
    args = {"site": "North", "telescope": "LSTN-01"}
    output_path = io_handler.get_output_directory(label="reports", sub_dir="parameters")
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

    read_parameters.produce_model_parameter_reports()

    file_path = output_path / args["telescope"] / "quantum_efficiency.md"
    assert file_path.exists()


def test__convert_to_md(telescope_model_lst, io_handler, db_config):
    args = {
        "telescope": telescope_model_lst.name,
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
    }
    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)
    parameter_name = "pm_photoelectron_spectrum"

    # testing with invalid file
    with pytest.raises(FileNotFoundError, match="Data file not found: "):
        read_parameters._convert_to_md(parameter_name, "1.0.0", "invalid-file.dat")

    # testing with valid file
    new_file = read_parameters._convert_to_md(
        parameter_name, "1.0.0", "tests/resources/spe_LST_2022-04-27_AP2.0e-4.dat"
    )
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
    input_path = Path("tests/resources/spe_LST_2022-04-27_AP2.0e-4.dat")
    with input_path.open("r", encoding="utf-8") as original_file:
        expected_lines = original_file.read().splitlines()[:30]
        expected_block = "\n".join(expected_lines)

    assert code_block.strip() == expected_block.strip()

    # testing with non-utf-8 file
    new_file = read_parameters._convert_to_md(
        parameter_name, "1.0.0", "tests/resources/example_non_utf-8_file.lis"
    )
    assert isinstance(new_file, str)
    assert Path(output_path / new_file).exists()


def test__generate_plots(tmp_path, db_config):
    args = {"telescope": "LSTN-design", "site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)
    input_file = tmp_path / "dummy_param.dat"
    input_file.write_text("dummy content")

    # Test case for parameter other than "camera_config_file"
    with patch.object(
        read_parameters, "_plot_parameter_tables", return_value=["plot2"]
    ) as mock_plot:
        result = read_parameters._generate_plots("some_param", "1.0.0", input_file, tmp_path, False)
        assert result == ["plot2"]
        mock_plot.assert_called_once()

    # Test case for parameter "camera_config_file"
    with patch.object(
        read_parameters, "_plot_camera_config", return_value=["camera_plot"]
    ) as mock_camera_plot:
        result = read_parameters._generate_plots(
            "camera_config_file", "1.0.0", input_file, tmp_path, False
        )
        assert result == ["camera_plot"]
        mock_camera_plot.assert_called_once()


def test__plot_camera_config_no_parameter_version(tmp_path, db_config):
    args = {"telescope": "LSTN-01", "site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)
    result = read_parameters._plot_camera_config("camera_config_file", None, tmp_path, False)
    assert result == []


def test__plot_parameter_tables(tmp_path, db_config):
    args = {"telescope": "LSTN-design", "site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)
    result = read_parameters._plot_parameter_tables(
        "pm_photoelectron_spectrum", "1.0.0", tmp_path, True
    )
    assert result == ["pm_photoelectron_spectrum_1.0.0_North_LSTN-design"]

    args = {"telescope": None, "site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)
    result = read_parameters._plot_parameter_tables("camera_config_file", "1.0.0", tmp_path, False)
    assert result == []


def test__format_parameter_value(io_handler, db_config):
    output_path = io_handler.get_output_directory()
    read_parameters = ReadParameters(db_config=db_config, args={}, output_path=output_path)
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


def test__wrap_at_underscores(io_handler, db_config):
    output_path = io_handler.get_output_directory()
    read_parameters = ReadParameters(db_config=db_config, args={}, output_path=output_path)

    # "this_is_a_test" -> parts: ['this', 'is', 'a', 'test']
    # builds: "this" (4), "this_is" (7), "this_is_a" (9), "this_is_a_test" (14) > 10 -> wrap
    # before "test"
    result_1 = read_parameters._wrap_at_underscores("this_is_a_test", 10)
    assert result_1 == "this_is_a test"

    result_2 = read_parameters._wrap_at_underscores("this_is_a_really_long_test", 10)
    assert result_2 == "this_is_a really long_test"

    # No underscores -> nothing to wrap
    result_3 = read_parameters._wrap_at_underscores("simpletext", 10)
    assert result_3 == "simpletext"

    # Whole string fits under max width
    result_4 = read_parameters._wrap_at_underscores("this_is_short", 20)
    assert result_4 == "this_is_short"

    result_5 = read_parameters._wrap_at_underscores("this_is_exactly_10", 10)
    assert result_5 == "this_is exactly_10"


def test__group_model_versions_by_parameter_version(io_handler, db_config):
    output_path = io_handler.get_output_directory()
    read_parameters = ReadParameters(db_config=db_config, args={}, output_path=output_path)

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
                "model_version": "6.0.0",
            },
            {
                "value": "all: 0.238006 GHz",
                "parameter_version": "1.0.0",
                "file_flag": False,
                "model_version": "5.0.0",
            },
        ],
        "pm_gain_index": [
            {
                "value": "4.5",
                "parameter_version": "1.0.0",
                "file_flag": False,
                "model_version": "6.0.0, 5.0.0",
            }
        ],
    }

    result = read_parameters._group_model_versions_by_parameter_version(mock_data)

    assert result == expected


def test__compare_parameter_across_versions(io_handler, db_config):
    args = {"site": "North", "telescope": "LSTN-01"}
    output_path = io_handler.get_output_directory(
        label="reports", sub_dir=f"parameters/{args['telescope']}"
    )
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

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
    qe_comparison = comparison_data.get("quantum_efficiency")
    assert qe_comparison["parameter_version" == "1.0.0"]["model_version"] == "6.0.0, 5.0.0"

    position_comparison = comparison_data.get("array_element_position_ground")
    assert position_comparison[0]["model_version"] != position_comparison[1]["model_version"]
    assert position_comparison["parameter_version" == "2.0.0"]["model_version"] == "6.0.0"

    assert len(comparison_data.get("only_prod6_param")) == 1
    assert "none_valued_param" not in comparison_data


def test__compare_parameter_across_versions_empty_param_dict(io_handler, db_config):
    """Test _compare_parameter_across_versions with empty parameter dictionaries."""
    args = {"site": "North", "telescope": "LSTN-01"}
    output_path = io_handler.get_output_directory(
        label="reports", sub_dir=f"parameters/{args['telescope']}"
    )
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

    # Test data with empty parameter dictionary for version 5.0.0
    mock_data = {
        "5.0.0": {},  # Empty parameter dictionary
        "6.0.0": {
            "quantum_efficiency": {
                "instrument": "LSTN-01",
                "site": "North",
                "parameter_version": "1.0.0",
                "value": QE_FILE_NAME,
                "unit": None,
                "file": True,
            }
        },
    }

    comparison_data = read_parameters._compare_parameter_across_versions(
        mock_data, ["quantum_efficiency"]
    )

    # Verify that data from version 6.0.0 is still processed
    assert "quantum_efficiency" in comparison_data
    qe_comparison = comparison_data["quantum_efficiency"]
    assert len(qe_comparison) == 1
    assert qe_comparison[0]["model_version"] == "6.0.0"
    assert qe_comparison[0]["parameter_version"] == "1.0.0"


def test_get_array_element_parameter_data_none_value(io_handler, db_config, mocker):
    """Test that get_array_element_parameter_data correctly handles None values."""
    args = {
        "telescope": "tel",
        "site": "North",
        "model_version": "v1",
    }
    output_path = io_handler.get_output_directory(sub_dir=f"{args['model_version']}")
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

    # Mock the model_parameters that will be filtered through
    model_params = {"test_param": "description"}
    mocker.patch("simtools.utils.names.model_parameters", return_value=model_params)

    # Mock all_parameter_data with a None value
    all_parameter_data = {
        "test_param": {
            "value": None,  # This will cause value_data to be None
            "instrument": "tel",
            "unit": "m",
        }
    }

    # Mock the TelescopeModel
    mock_telescope_model = mocker.MagicMock()
    mock_telescope_model.name = "tel"
    mock_telescope_model.site = args["site"]
    mock_telescope_model.model_version = args["model_version"]

    # Mock get_model_parameters to return our all_parameter_data
    mocker.patch.object(read_parameters.db, "get_model_parameters", return_value=all_parameter_data)

    # Mock get_all_parameter_descriptions
    mocker.patch.object(
        read_parameters,
        "get_all_parameter_descriptions",
        return_value=(
            {
                "test_param": {
                    "description": DESCRIPTION,
                    "short_description": "Test",
                    "inst_class": "Structure",
                }
            }
        ),
    )

    result = read_parameters.get_array_element_parameter_data(mock_telescope_model)

    # Assert result is empty (parameter was skipped due to None value)
    assert len(result) == 0

    # Verify the mocks were called
    assert read_parameters.db.get_model_parameters.called


def test_produce_observatory_report(io_handler, db_config, mocker):
    """Test generation of observatory parameter report with all parameter types and empty data."""
    args = {"site": "North", "model_version": "6.0.0"}
    output_path = io_handler.get_output_directory()
    read_parameters = ReadParameters(db_config, args, output_path)

    # Test with empty parameter data
    mock_logger = mocker.patch("logging.Logger.warning")

    with patch.object(read_parameters.db, "get_model_parameters", return_value={}):
        read_parameters.produce_observatory_report()

        # Verify warning was logged
        mock_logger.assert_called_once_with(
            f"No observatory parameters found for site {args['site']}"
        )

        # Verify DB call was made with correct parameters
        read_parameters.db.get_model_parameters.assert_called_once_with(
            site=read_parameters.site,
            array_element_name=f"OBS-{read_parameters.site}",
            collection="sites",
            model_version=read_parameters.model_version,
        )

        # Verify no file was created
        output_file = Path(output_path) / f"OBS-{args['site']}.md"
        assert not output_file.exists()

    # Test with valid parameter data
    mock_parameters = {
        "site_elevation": {
            "value": 2200,
            "unit": "m",
            "parameter_version": "1.0",
        },
        "array_layouts": {
            "value": [
                {"name": "Layout1", "elements": ["LST1", "LST2"]},
                {"name": "Layout2", "elements": ["MST1", "MST2"]},
            ],
            "unit": None,
            "parameter_version": "1.0",
        },
        "array_triggers": {
            "value": [
                {
                    "name": "LSTN_array",
                    "multiplicity": {"value": 2, "unit": None},
                    "width": {"value": 120, "unit": "ns"},
                    "hard_stereo": {"value": True, "unit": None},
                    "min_separation": {"value": None, "unit": "m"},
                },
                {
                    "name": "MSTN_array",
                    "multiplicity": {"value": 2, "unit": None},
                    "width": {"value": 200, "unit": "ns"},
                    "hard_stereo": {"value": False, "unit": None},
                    "min_separation": {"value": 40, "unit": "m"},
                },
            ],
            "unit": None,
            "parameter_version": "1.0",
        },
        "none_valued_param": {
            "value": None,
            "unit": None,
            "parameter_version": "1.0",
        },
    }

    with patch.object(read_parameters.db, "get_model_parameters", return_value=mock_parameters):
        read_parameters.produce_observatory_report()

        # Verify DB call was made with correct parameters
        read_parameters.db.get_model_parameters.assert_called_once_with(
            site=read_parameters.site,
            array_element_name=f"OBS-{read_parameters.site}",
            collection="sites",
            model_version=read_parameters.model_version,
        )

        # Check output file exists and contains expected content
        output_file = Path(output_path) / f"OBS-{args['site']}.md"
        assert output_file.exists()

        content = output_file.read_text()
        assert "# Observatory Parameters" in content
        assert "| Parameter | Value |" in content
        assert "| site_elevation | 2200 m |" in content
        assert "none_valued_param" not in content


def test__write_array_layouts_section(io_handler, db_config, mocker):
    """Test writing array layouts section."""
    args = {"site": "North", "model_version": "6.0.0"}
    output_path = io_handler.get_output_directory()
    read_parameters = ReadParameters(db_config, args, output_path)

    mock_layouts = [
        {
            "name": "Layout1",
            "elements": ["LST1", "LST2", "MST1"],
        },
        {
            "name": "Layout2",
            "elements": ["LST1", "MST1", "MST2"],
        },
    ]

    with StringIO() as file:
        read_parameters._write_array_layouts_section(file, mock_layouts)
        output = file.getvalue()

    # Verify section header
    assert "## Array Layouts" in output

    # Verify layout names
    assert "### Layout1" in output
    assert "[MST1](MST1.md)" in output
    assert "### Layout2" in output
    assert "[MST2](MST2.md)" in output

    # Verify image links
    assert "![Layout1 Layout](/_images/OBS-North_Layout1_6-0-0.png)" in output


def test__write_array_triggers_section(io_handler, db_config):
    """Test writing array triggers section."""
    args = {}
    output_path = io_handler.get_output_directory()
    read_parameters = ReadParameters(db_config, args, output_path)

    mock_triggers = [
        {
            "name": "Trigger1",
            "multiplicity": {"value": 2, "unit": None},
            "width": {"value": 100, "unit": "ns"},
            "hard_stereo": {"value": True, "unit": None},
            "min_separation": {"value": 50, "unit": "m"},
        },
        {
            "name": "Trigger2",
            "multiplicity": {"value": 3, "unit": "telescopes"},
            "width": {"value": 150, "unit": "ns"},
            "hard_stereo": {"value": False, "unit": None},
            "min_separation": {"value": 75, "unit": "m"},
        },
    ]

    with StringIO() as file:
        read_parameters._write_array_triggers_section(file, mock_triggers)
        output = file.getvalue()

    # Verify section header and table headers
    assert "## Array Trigger Configurations" in output
    assert "| Trigger Name | Multiplicity | Width | Hard Stereo | Min Separation |" in output

    # Verify trigger data
    assert "| Trigger1 | 2 | 100 ns | Yes | 50 m |" in output
    assert "| Trigger2 | 3 telescopes | 150 ns | No | 75 m |" in output


def test__write_parameters_table(io_handler, db_config):
    """Test writing parameters table."""
    args = {}
    output_path = io_handler.get_output_directory()
    read_parameters = ReadParameters(db_config, args, output_path)

    mock_params = {
        "site_elevation": {"value": 2200, "unit": "m", "parameter_version": "1.0.0"},
        "array_layouts": {"value": [], "unit": None, "parameter_version": "2.0.0"},
        "array_triggers": {"value": [], "unit": None, "parameter_version": "3.0.0"},
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


def test_model_version_setter_with_valid_string(db_config, io_handler):
    """Test setting model_version with a valid string."""
    args = {"model_version": "6.0.0"}
    output_path = io_handler.get_output_directory()
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

    read_parameters.model_version = "7.0.0"
    assert read_parameters.model_version == "7.0.0"


def test_model_version_setter_with_invalid_list(db_config, io_handler):
    """Test setting model_version with an invalid list containing more than one element."""
    args = {"model_version": "6.0.0"}
    output_path = io_handler.get_output_directory()
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

    error_message = "Only one model version can be passed to ReadParameters, not a list."

    with pytest.raises(ValueError, match=error_message):
        read_parameters.model_version = ["7.0.0"]

    with pytest.raises(ValueError, match=error_message):
        read_parameters.model_version = ["7.0.0", "8.0.0"]

    with pytest.raises(ValueError, match=error_message):
        read_parameters.model_version = []


@pytest.mark.parametrize(
    ("simulation_software", "param_dict", "descriptions"),
    [
        (
            "corsika",
            {
                "corsika cherenkov photon bunch_size": {
                    "value": 5.0,
                    "unit": "",
                    "parameter_version": "1.0.0",
                },
                "corsika particle kinetic energy cutoff": {
                    "value": [0.3, 0.1, 0.02, 0.02],
                    "unit": "GeV",
                    "parameter_version": "1.0.0",
                },
                "none_value": {
                    "value": None,
                    "unit": "GeV",
                    "parameter_version": "1.0.0",
                },
            },
            {
                "corsika cherenkov photon bunch_size": {
                    "description": "Cherenkov bunch size definition.",
                    "short_description": "Bunch size",
                },
                "corsika particle kinetic energy cutoff": {
                    "description": "Kinetic energy cutoffs for different particle types.",
                    "short_description": "Energy cutoffs",
                },
                "none_value": {
                    "description": "None value parameter description.",
                    "short_description": None,
                },
            },
        ),
        (
            "sim_telarray",
            {
                "param1": {
                    "value": 5.0,
                    "unit": "",
                    "parameter_version": "1.0.0",
                },
                "param2": {
                    "value": [0.3, 0.1, 0.02, 0.02],
                    "unit": "GeV",
                    "parameter_version": "1.0.0",
                },
                "none_value": {
                    "value": None,
                    "unit": "GeV",
                    "parameter_version": "1.0.0",
                },
            },
            {
                "param1": {
                    "description": "Description 1",
                    "short_description": "Short description 1",
                },
                "param2": {
                    "description": "Description 2",
                    "short_description": "Short description 2",
                },
                "none_value": {
                    "description": "None value parameter description.",
                    "short_description": None,
                },
            },
        ),
    ],
)
def test_get_simulation_configuration_data(
    simulation_software, param_dict, descriptions, io_handler, db_config
):
    args = {
        "model_version": "6.0.0",
        "simulation_software": simulation_software,
    }
    output_path = io_handler.get_output_directory(sub_dir=f"{args['model_version']}")
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

    with (
        patch.object(read_parameters, "get_all_parameter_descriptions", return_value=descriptions),
        patch.object(read_parameters.db, "export_model_files") as mock_export,
        patch.object(
            read_parameters.db, "get_simulation_configuration_parameters", return_value=param_dict
        ),
    ):
        data = read_parameters.get_simulation_configuration_data()

        assert isinstance(data, list)

        if simulation_software == "corsika":
            assert len(data) == 2
            assert data[0][1] == "corsika cherenkov photon bunch_size"  # Parameter name
            assert data[0][2] == "1.0.0"  # Parameter version
            assert data[0][3] == "5.0"  # Value (formatted)
            assert data[0][5] == "Bunch size"  # Short description
            assert data[1][3] == "0.3 GeV, 0.1 GeV, 0.02 GeV, 0.02 GeV"
            mock_export.assert_called_once()

        elif simulation_software == "sim_telarray":
            assert len(data) > 0  # Ensure data is not empty
            assert data[0][0] == "LSTN-01"
            assert data[0][1] == "param1"  # Parameter name
            assert data[0][2] == "1.0.0"  # Parameter version
            assert data[0][3] == "5.0"  # Value (formatted)
            assert data[0][5] == "Short description 1"  # Short description
            assert data[1][3] == "0.3 GeV, 0.1 GeV, 0.02 GeV, 0.02 GeV"
            assert data[1][5] == "Short description 2"  # Short description


def test__write_to_file(telescope_model_lst, io_handler, db_config):
    args = {
        "telescope": telescope_model_lst.name,
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
        "simulation_software": "corsika",
    }
    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

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

    output_file = output_path / "output.md"

    with output_file.open("w") as f:
        read_parameters._write_to_file(mock_data, f)

    result = output_file.read_text()

    assert "| Parameter Name" in result
    assert "| param_1" in result
    assert "Short 1" in result
    assert "0.3 GeV, 0.2 GeV" in result


def test_produce_simulation_configuration_report(io_handler, db_config):
    args = {
        "telescope": "LSTN-01",
        "site": "North",
        "model_version": "6.0.0",
        "simulation_software": "sim_telarray",
    }
    output_path = io_handler.get_output_directory(
        label="reports", sub_dir=f"productions/{args.get('model_version')}"
    )
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

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

    with patch.object(read_parameters, "get_simulation_configuration_data", return_value=mock_data):
        read_parameters.produce_simulation_configuration_report()

        report_file = output_path / f"configuration_{read_parameters.software}.md"
        assert report_file.exists()

        content = report_file.read_text()

        assert f"# configuration_{read_parameters.software}" in content
        assert "| Parameter Name" in content
        assert "Buffer limits for input and output of eventio data." in content
        assert "mt19937" in content

    # testing for corsika
    args["simulation_software"] = "corsika"
    read_parameters_corsika = ReadParameters(
        db_config=db_config, args=args, output_path=output_path
    )

    with patch.object(
        read_parameters_corsika, "get_simulation_configuration_data", return_value=mock_data
    ):
        read_parameters_corsika.produce_simulation_configuration_report()

        report_file_corsika = output_path / f"configuration_{read_parameters.software}.md"
        assert report_file_corsika.exists()


def test_produce_calibration_reports(io_handler, db_config, mocker):
    """Test generation of calibration report for an array element."""
    args = {"model_version": "6.0.0"}
    output_path = io_handler.get_output_directory(sub_dir=f"{args['model_version']}")
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)
    description = "Description for laser events"
    # Mock array elements
    mock_array_elements = ["ILLN-01"]
    mocker.patch.object(read_parameters.db, "get_array_elements", return_value=mock_array_elements)

    mock_data = {
        "5.0.0": {},
        "6.0.0": {
            "dark_events": {
                "value": 0,
                "unit": None,
                "parameter_version": "1.0.0",
                "instrument": "ILLN-01",
                "collection": "telescopes",
            },
            "laser_events": {
                "value": 10,
                "unit": None,
                "parameter_version": "1.0.0",
                "instrument": "ILLN-design",
                "collection": "calibration_devices",
            },
            "pedestal_events": {
                "value": 100,
                "unit": None,
                "parameter_version": "1.0.0",
                "instrument": "ILLN-01",
                "collection": "calibration_devices",
            },
            "array_element_position_ground": {
                "value": [0.0, 0.0, 0.0],
                "unit": "m",
                "parameter_version": "1.0.0",
                "instrument": "ILLN-01",
                "collection": "calibration_devices",
            },
        },
    }

    # Mock get_model_parameters
    mocker.patch.object(
        read_parameters.db,
        "get_model_parameters",
        return_value=mock_data[args.get("model_version")],
    )

    # Mock descriptions
    mock_calib_descriptions = {
        "dark_events": {
            "description": "Dark pedestal events",
            "short_description": "Dark events",
            "inst_class": "Calibration",
        },
        "laser_events": {
            "description": description,
            "short_description": None,
            "inst_class": "Calibration",
        },
        "pedestal_events": {
            "description": "Pedestal events with open lid",
            "short_description": "Pedestal events",
            "inst_class": "Camera",
        },
    }

    mock_position_descriptions = {
        "array_element_position_ground": {
            "description": "Position of the telescope",
            "short_description": "Position",
            "inst_class": "Structure",
        }
    }

    with patch.object(
        read_parameters,
        "get_all_parameter_descriptions",
        side_effect=lambda collection: mock_position_descriptions
        if collection == "telescopes"
        else mock_calib_descriptions,
    ) as mock_desc:
        result = read_parameters.get_calibration_data(
            mock_data[args.get("model_version")], "ILLN-01"
        )

        # Check that descriptions were fetched for both collections
        assert mock_desc.call_count == 2

        # Verify the structure and ordering of the result
        assert len(result) == 4

        # Verify the content of a specific entry
        laser_event = next(x for x in result if x[1] == "laser_events")
        assert laser_event[2] == "1.0.0"  # parameter version
        assert laser_event[3] == "10"  # value
        assert laser_event[4] == description  # description
        assert laser_event[5] == description  # short description set to description when None

        # Run the method
        read_parameters.produce_calibration_reports()

    # Verify output file exists
    output_file = Path(output_path) / "ILLN-01.md"
    assert output_file.exists()

    # Check file content
    content = output_file.read_text()

    assert "# ILLN-01" in content
    assert "## Calibration" in content
    assert "| Values" in content
    assert "| Short Description" in content
    assert "1.0.0" in content
    assert "| laser events |" in content

    # Check comparison reports
    output_path_2 = Path(output_path).parent.parent / "parameters"
    output_file_2 = output_path_2 / "ILLN-01/array_element_position_ground.md"
    assert output_file_2.exists()
    content_2 = output_file_2.read_text()
    assert "# array_element_position_ground" in content_2
    assert "**Telescope**: ILLN-01" in content_2


def test_get_calibration_data(io_handler, db_config):
    args = {
        "model_version": "6.0.0",
    }
    output_path = io_handler.get_output_directory(sub_dir=f"{args.get('model_version')}")
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

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
        "pedestal_events": {
            "value": 100,
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
    }

    # Mock descriptions
    mock_descriptions = {
        "dark_events": {
            "description": "Dark pedestal events",
            "short_description": "Dark events",
            "inst_class": "Calibration",
        },
        "laser_events": {
            "description": "Laser calibration events",
            "short_description": None,
            "inst_class": "Calibration",
        },
        "pedestal_events": {
            "description": "Pedestal events with open lid",
            "short_description": "Pedestal events",
            "inst_class": "Camera",
        },
        "none_value": {
            "description": "None value parameter description.",
            "short_description": "Short description for none value",
            "inst_class": "Calibration",
        },
    }

    # Mock descriptions using patch
    with patch.object(read_parameters, "get_all_parameter_descriptions") as mock_desc:
        mock_desc.return_value = mock_descriptions
        result = read_parameters.get_calibration_data(mock_data, "ILLN-01")

    # Assert the result contains the expected data
    assert result[0][0] == "Camera"
    assert len(result[0]) == 6
    assert len(result) == 3


class DummyTelescope:
    def __init__(self, site, name, model_version, param_versions=None):
        self.site = site
        self.name = name
        self.model_version = model_version
        self._param_versions = param_versions or {}

    def get_parameter_version(self, parameter):
        return self._param_versions.get(parameter)


def test_get_array_element_parameter_data_simple(tmp_path, monkeypatch):
    """Simple test for get_array_element_parameter_data using mocked DB calls."""

    # Prepare ReadParameters with a harmless config and an output path
    args = {"telescope": "LSTN-01", "site": "North", "model_version": "1.0.0", "observatory": None}
    rp = ReadParameters(db_config=None, args=args, output_path=tmp_path)

    # Replace the db on the instance with a mock to avoid any DB access
    rp.db = Mock()

    # Create a minimal set of parameter data returned by the DB
    all_parameter_data = {
        "test_param": {
            "unit": "m",
            "value": 42,
            "parameter_version": "1.0.0",
            "file": False,
            "instrument": "OTHER",  # different from telescope name to avoid bolding
        }
    }

    rp.db.get_model_parameters.return_value = all_parameter_data
    rp.db.export_model_files.return_value = None

    # Mock parameter descriptions that get_array_element_parameter_data expects
    rp.get_all_parameter_descriptions = Mock(
        return_value={
            "test_param": {
                "description": DESCRIPTION,
                "short_description": SHORT_DESC,
                "inst_class": "Telescope",
            }
        }
    )

    # Dummy telescope that provides parameter versions
    tel = DummyTelescope(
        site="North",
        name="LSTN-01",
        model_version="1.0.0",
        param_versions={"test_param": "1.0.0"},
    )

    # Patch names to expose our test_param and avoid reading resource files
    monkeypatch.setattr(names, "model_parameters", lambda *args, **kwargs: {"test_param": {}})
    monkeypatch.setattr(names, "is_design_type", lambda _name: False)

    data = rp.get_array_element_parameter_data(telescope_model=tel, collection="telescopes")

    # Expect one row with formatted value '42 m'
    assert data == [["Telescope", "test_param", "1.0.0", "42 m", DESCRIPTION, "Short"]]


def test_get_array_element_parameter_data_instrument_specific(tmp_path, monkeypatch):
    """Test that instrument-specific parameters are wrapped in bold/italic markers."""

    args = {"telescope": "LSTN-01", "site": "North", "model_version": "1.0.0", "observatory": None}
    rp = ReadParameters(db_config=None, args=args, output_path=tmp_path)
    rp.db = Mock()

    all_parameter_data = {
        "test_param": {
            "unit": "m",
            "value": 42,
            "parameter_version": "1.0.0",
            "file": False,
            "instrument": "LSTN-01",
        }
    }

    rp.db.get_model_parameters.return_value = all_parameter_data
    rp.db.export_model_files.return_value = None

    rp.get_all_parameter_descriptions = Mock(
        return_value={
            "test_param": {
                "description": DESCRIPTION,
                "short_description": SHORT_DESC,
                "inst_class": "Telescope",
            }
        }
    )

    tel = DummyTelescope(
        site="North",
        name="LSTN-01",
        model_version="1.0.0",
        param_versions={"test_param": "1.0.0"},
    )

    monkeypatch.setattr(names, "model_parameters", lambda *args, **kwargs: {"test_param": {}})
    monkeypatch.setattr(names, "is_design_type", lambda _name: False)

    data = rp.get_array_element_parameter_data(telescope_model=tel, collection="telescopes")

    assert data == [
        [
            "Telescope",
            "***test_param***",
            "***1.0.0***",
            "***42 m***",
            f"***{DESCRIPTION}***",
            f"***{SHORT_DESC}***",
        ]
    ]


def test_get_array_element_parameter_data_file_parameter(tmp_path, monkeypatch):
    """Test that file parameters are converted to markdown links using _convert_to_md."""

    args = {"telescope": "LSTN-01", "site": "North", "model_version": "1.0.0", "observatory": None}
    rp = ReadParameters(db_config=None, args=args, output_path=tmp_path)
    rp.db = Mock()

    # Parameter with file flag set
    all_parameter_data = {
        "file_param": {
            "unit": None,
            "value": "myfile.dat",
            "parameter_version": "1.0.0",
            "file": True,
            "instrument": "OTHER",
        }
    }

    rp.db.get_model_parameters.return_value = all_parameter_data
    # export_model_files is called by get_array_element_parameter_data; mock it
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

    # Monkeypatch _convert_to_md to avoid file IO and return a relative path
    def _fake_convert_to_md(self, parameter, parameter_version, input_file, design_type):
        return "_data_files/myfile.md"

    monkeypatch.setattr(ReadParameters, "_convert_to_md", _fake_convert_to_md)

    tel = DummyTelescope(
        site="North",
        name="LSTN-01",
        model_version="1.0.0",
        param_versions={"file_param": "1.0.0"},
    )

    monkeypatch.setattr(names, "model_parameters", lambda *args, **kwargs: {"file_param": {}})
    monkeypatch.setattr(names, "is_design_type", lambda _name: False)

    data = rp.get_array_element_parameter_data(telescope_model=tel, collection="telescopes")

    # Expect the value to be a markdown link using the returned relative path
    assert data == [
        [
            "Telescope",
            "file_param",
            "1.0.0",
            "[myfile.dat](_data_files/myfile.md)",
            DESCRIPTION,
            SHORT_DESC,
        ]
    ]


def test_plot_camera_config(tmp_path, db_config, mocker):
    args = {"telescope": "LSTN-01", "site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)

    # Mock input file and output path
    input_file = tmp_path / "camera_config_file.dat"
    input_file.touch()
    plot_name = input_file.stem.replace(".", "-")
    plot_path = tmp_path / f"{plot_name}.png"

    # Mock plot_pixels.plot to avoid actual plotting
    mock_plot = mocker.patch("simtools.visualization.plot_pixels.plot")

    # Test when plot does not exist
    result = read_parameters._plot_camera_config(
        "camera_config_file", "1.0.0", input_file, tmp_path
    )
    assert result == [plot_name]
    mock_plot.assert_called_once_with(
        config={
            "file_name": input_file.name,
            "telescope": args["telescope"],
            "parameter_version": "1.0.0",
            "site": args["site"],
            "model_version": args["model_version"],
            "parameter": "camera_config_file",
        },
        output_file=plot_path.with_suffix(""),
        db_config=db_config,
    )

    # Test when plot already exists
    plot_path.touch()
    mock_plot.reset_mock()
    result = read_parameters._plot_camera_config(
        "camera_config_file", "1.0.0", input_file, tmp_path
    )
    assert result == [plot_name]
    mock_plot.assert_not_called()
