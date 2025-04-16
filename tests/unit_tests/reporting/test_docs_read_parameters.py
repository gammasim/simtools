import re
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import astropy.units as u
import pytest

from simtools.reporting.docs_read_parameters import ReadParameters

# Test constants
QE_FILE_NAME = "qe_lst1_20200318_high+low.dat"


def test_get_all_parameter_descriptions(telescope_model_lst, io_handler, db_config):
    args = {
        "telescope": telescope_model_lst.name,
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
    }
    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)
    # Call get_all_parameter_descriptions
    descriptions, short_descriptions, inst_class = read_parameters.get_all_parameter_descriptions()

    assert isinstance(descriptions.get("focal_length"), str)
    assert isinstance(short_descriptions.get("focal_length"), str)
    assert isinstance(inst_class.get("focal_length"), str)


def test_get_array_element_parameter_data(telescope_model_lst, io_handler, db_config):
    args = {
        "telescope": telescope_model_lst.name,
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
    }
    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

    result = read_parameters.get_array_element_parameter_data(telescope_model_lst)

    # Assert the result contains the expected data
    if result[1] == "focal_length":
        assert result[0] == "Structure"
        assert result[3] == (2800.0 * u.cm)
        assert result[4] == "Nominal overall focal length of the entire telescope."


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
    parameter_name = "test"

    # testing with invalid file
    with pytest.raises(FileNotFoundError, match="Data file not found: "):
        read_parameters._convert_to_md(parameter_name, "invalid-file.dat")

    # testing with valid file
    new_file = read_parameters._convert_to_md(
        parameter_name, "tests/resources/spe_LST_2022-04-27_AP2.0e-4.dat"
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
        parameter_name, "tests/resources/example_non_utf-8_file.lis"
    )
    assert isinstance(new_file, str)
    assert Path(output_path / new_file).exists()


def test__format_parameter_value(io_handler, db_config):
    output_path = io_handler.get_output_directory()
    read_parameters = ReadParameters(db_config=db_config, args={}, output_path=output_path)
    parameter_name = "test"

    mock_data_1 = [[24.74, 9.0, 350.0, 1066.0], ["ns", "ns", "V", "V"], False]
    result_1 = read_parameters._format_parameter_value(parameter_name, *mock_data_1)
    assert result_1 == "24.74 ns, 9.0 ns, 350.0 V, 1066.0 V"

    mock_data_2 = [4.0, " ", None]
    result_2 = read_parameters._format_parameter_value(parameter_name, *mock_data_2)
    assert result_2 == "4.0"

    mock_data_3 = [
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        "GHz",
        False,
    ]
    result_3 = read_parameters._format_parameter_value(parameter_name, *mock_data_3)
    assert result_3 == "all: 0.2 GHz"

    mock_data_4 = [[1, 2, 3, 4], "m", None]
    result_4 = read_parameters._format_parameter_value(parameter_name, *mock_data_4)
    assert result_4 == "1 m, 2 m, 3 m, 4 m"


def test__wrap_at_underscores(io_handler, db_config):
    output_path = io_handler.get_output_directory()
    read_parameters = ReadParameters(db_config=db_config, args={}, output_path=output_path)

    # "this_is_a_test" -> parts: ['this', 'is', 'a', 'test']
    # builds: "this" (4), "this_is" (7), "this_is_a" (9), "this_is_a_test" (14) > 10 -> wrap before "test"
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
            {"test_param": "Test parameter"},
            {"test_param": "Test"},
            {"test_param": "Structure"},
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
    assert "![Layout1 Layout](../../_images/OBS-North_Layout1_6-0-0.png)" in output


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


def test_get_simulation_configuration_data(telescope_model_lst, io_handler, db_config):
    args = {
        "telescope": telescope_model_lst.name,
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
        "simulation_software": "corsika",
    }
    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

    mock_param_dict = {
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
    }

    mock_descriptions = (
        {
            "corsika cherenkov photon bunch_size": "Cherenkov bunch size definition.",
            "corsika particle kinetic energy cutoff": "Kinetic energy cutoffs for different particle types.",
            "none_value": "None value parameter description.",
        },
        {
            "corsika cherenkov photon bunch_size": "Bunch size",
            "corsika particle kinetic energy cutoff": "Energy cutoffs",
            "none_value": None,
        },
    )

    with (
        patch.object(
            read_parameters, "get_all_parameter_descriptions", return_value=mock_descriptions
        ),
        patch.object(read_parameters.db, "export_model_files") as mock_export,
        patch.object(
            read_parameters.db,
            "get_simulation_configuration_parameters",
            return_value=mock_param_dict,
        ),
    ):
        data = read_parameters.get_simulation_configuration_data()

        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0][0] == "corsika cherenkov photon bunch_size"
        assert data[0][1] == "1.0.0"
        assert data[0][2] == "5.0"
        assert data[0][4] == "Bunch size"
        assert data[1][2] == "0.3 GeV, 0.1 GeV, 0.02 GeV, 0.02 GeV"
        mock_export.assert_called_once()


def test_produce_simulation_configuration_report(io_handler, db_config):
    args = {
        "telescope": "LSTN-01",
        "site": "North",
        "model_version": "6.0.0",
        "simulation_software": "simtel",
    }
    output_path = io_handler.get_output_directory(
        label="reports", sub_dir=f"productions/{args.get('model_version')}"
    )
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

    mock_data = [
        (
            "iobuf maximum",
            "1.0.0",
            "100 byte",
            "Description",
            "Buffer limits for input and output of eventio data.",
        ),
        (
            "random generator",
            "1.0.0",
            "mt19937",
            "Random generator used.",
            None,
        ),
    ]

    with patch.object(read_parameters, "get_simulation_configuration_data", return_value=mock_data):
        read_parameters.produce_simulation_configuration_report()

        report_file = (
            output_path
            / f"{read_parameters.array_element}_configuration_{read_parameters.software}.md"
        )
        assert report_file.exists()

        content = report_file.read_text()

        assert f"# configuration_{read_parameters.software}" in content
        assert "| Parameter Name" in content
        assert "Buffer limits for input and output of eventio data." in content
        assert "mt19937" in content
