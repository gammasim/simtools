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


def test_get_all_parameter_descriptions(telescope_model_lst, db_config, tmp_path):
    args = {
        "telescope": telescope_model_lst.name,
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
    }
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)
    # Call get_all_parameter_descriptions
    description_dict = read_parameters.get_all_parameter_descriptions()

    assert isinstance(description_dict.get("focal_length"), dict)
    assert isinstance(description_dict.get("focal_length").get("description"), str)
    assert isinstance(description_dict.get("focal_length").get("short_description"), str)
    assert isinstance(description_dict.get("focal_length").get("inst_class"), str)


def test_produce_array_element_report(telescope_model_lst, db_config, tmp_path):
    """Test array element report generation with both observatory and telescope scenarios."""
    # Test observatory report path
    args = {
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
        "observatory": True,
    }
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)

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
        obs_file = tmp_path / f"OBS-{args['site']}.md"
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
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)

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
        tel_file = tmp_path / f"{telescope_model_lst.name}.md"
        assert tel_file.exists()

        # Verify DB was called with correct parameters
        read_parameters.db.get_model_parameters.assert_called_once_with(
            site=args["site"],
            array_element_name=args["telescope"],
            collection="telescopes",
            model_version=args["model_version"],
        )


def test_produce_model_parameter_reports(db_config, tmp_test_directory):
    args = {"site": "North", "telescope": "LSTN-01"}
    output_path = tmp_test_directory
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

    read_parameters.produce_model_parameter_reports()

    file_path = output_path / args["telescope"] / "quantum_efficiency.md"
    assert file_path.exists()


def test__convert_to_md(telescope_model_lst, db_config, tmp_test_directory):
    args = {
        "telescope": telescope_model_lst.name,
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
    }
    output_path = tmp_test_directory
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)
    parameter_name = "pm_photoelectron_spectrum"

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


def test__generate_plots(tmp_test_directory, db_config):
    args = {"telescope": "LSTN-design", "site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_test_directory)
    input_file = Path(tmp_test_directory / "dummy_param.dat")
    input_file.write_text("dummy content")

    # Test case for parameter other than "camera_config_file"
    with patch.object(
        read_parameters, "_plot_parameter_tables", return_value=["plot2"]
    ) as mock_plot:
        result = read_parameters._generate_plots(
            "some_param", "1.0.0", input_file, tmp_test_directory
        )
        assert result == ["plot2"]
        mock_plot.assert_called_once()

    # Test case for parameter "camera_config_file"
    with patch.object(
        read_parameters, "_plot_camera_config", return_value=["camera_plot"]
    ) as mock_camera_plot:
        result = read_parameters._generate_plots(
            "camera_config_file", "1.0.0", input_file, tmp_test_directory
        )
        assert result == ["camera_plot"]
        mock_camera_plot.assert_called_once()


def test__plot_camera_config_no_parameter_version(tmp_test_directory, db_config):
    args = {"telescope": "LSTN-01", "site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_test_directory)
    result = read_parameters._plot_camera_config(
        "camera_config_file", None, tmp_test_directory, False
    )
    assert result == []


def test__plot_parameter_tables(tmp_test_directory, db_config):
    args = {"telescope": "LSTN-design", "site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_test_directory)
    result = read_parameters._plot_parameter_tables(
        "pm_photoelectron_spectrum", "1.0.0", Path(tmp_test_directory)
    )
    assert result == ["pm_photoelectron_spectrum_1.0.0_North_LSTN-design"]

    args = {"telescope": None, "site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_test_directory)
    result = read_parameters._plot_parameter_tables(
        "camera_config_file", "1.0.0", Path(tmp_test_directory)
    )
    assert result == []


def test__format_parameter_value(db_config, tmp_path):
    read_parameters = ReadParameters(db_config=db_config, args={}, output_path=tmp_path)
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


def test__group_model_versions_by_parameter_version(db_config, tmp_path):
    read_parameters = ReadParameters(db_config=db_config, args={}, output_path=tmp_path)

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
                "model_version": "5.0.0, 6.0.0",
            }
        ],
    }

    result = read_parameters._group_model_versions_by_parameter_version(mock_data)

    assert result == expected


def test__compare_parameter_across_versions(tmp_test_directory, db_config):
    args = {"site": "North", "telescope": "LSTN-01"}
    output_path = tmp_test_directory
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


def test__compare_parameter_across_versions_empty_param_dict(db_config, tmp_path):
    """Test _compare_parameter_across_versions with empty parameter dictionaries."""
    args = {"site": "North", "telescope": "LSTN-01"}
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)

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


def test_get_array_element_parameter_data_none_value(db_config, mocker, tmp_path):
    """Test that get_array_element_parameter_data correctly handles None values."""
    args = {
        "telescope": "tel",
        "site": "North",
        "model_version": "v1",
    }
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)

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


def test_produce_observatory_report(db_config, mocker, tmp_path):
    """Test generation of observatory parameter report with all parameter types and empty data."""
    args = {"site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(db_config, args, tmp_path)

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
        output_file = Path(tmp_path) / f"OBS-{args['site']}.md"
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
        output_file = Path(tmp_path) / f"OBS-{args['site']}.md"
        assert output_file.exists()

        content = output_file.read_text()
        assert "# Observatory Parameters" in content
        assert "| Parameter | Value |" in content
        assert "| site_elevation | 2200 m |" in content
        assert "none_valued_param" not in content


def test__write_array_layouts_section(db_config, tmp_path):
    """Test writing array layouts section."""
    args = {"site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(db_config, args, tmp_path)

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


def test__write_array_triggers_section(db_config, tmp_path):
    """Test writing array triggers section."""
    args = {}
    read_parameters = ReadParameters(db_config, args, tmp_path)

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


def test__write_parameters_table(db_config, tmp_path):
    """Test writing parameters table."""
    args = {}
    read_parameters = ReadParameters(db_config, args, tmp_path)

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


def test_model_version_setter_with_valid_string(db_config, tmp_path):
    """Test setting model_version with a valid string."""
    args = {"model_version": "6.0.0"}
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)

    read_parameters.model_version = "7.0.0"
    assert read_parameters.model_version == "7.0.0"


def test_model_version_setter_with_invalid_list(db_config, tmp_path):
    """Test setting model_version with an invalid list containing more than one element."""
    args = {"model_version": "6.0.0"}
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)

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
    simulation_software, param_dict, descriptions, db_config, tmp_path
):
    args = {
        "model_version": "6.0.0",
        "simulation_software": simulation_software,
    }
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)

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


def test__write_to_file(telescope_model_lst, db_config, tmp_path):
    args = {
        "telescope": telescope_model_lst.name,
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
        "simulation_software": "corsika",
    }
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)

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


def test_produce_simulation_configuration_report(db_config, tmp_path):
    args = {
        "telescope": "LSTN-01",
        "site": "North",
        "model_version": "6.0.0",
        "simulation_software": "sim_telarray",
    }
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)

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

        report_file = tmp_path / f"configuration_{read_parameters.software}.md"
        assert report_file.exists()

        content = report_file.read_text()

        assert f"# configuration_{read_parameters.software}" in content
        assert "| Parameter Name" in content
        assert "Buffer limits for input and output of eventio data." in content
        assert "mt19937" in content

    # testing for corsika
    args["simulation_software"] = "corsika"
    read_parameters_corsika = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)

    with patch.object(
        read_parameters_corsika, "get_simulation_configuration_data", return_value=mock_data
    ):
        read_parameters_corsika.produce_simulation_configuration_report()

        report_file_corsika = tmp_path / f"configuration_{read_parameters.software}.md"
        assert report_file_corsika.exists()


def test_produce_calibration_reports(db_config, mocker, tmp_path):
    """Test generation of calibration report for an array element."""
    args = {"model_version": "6.0.0"}
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)
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
        assert len(result) == 3

        # Verify the content of a specific entry
        laser_event = next(x for x in result if x[1] == "laser_events")
        assert laser_event[2] == "1.0.0"  # parameter version
        assert laser_event[3] == "10"  # value
        assert laser_event[4] == description  # description
        assert laser_event[5] == description  # short description set to description when None

        # Run the method
        read_parameters.produce_calibration_reports()

    # Verify output file exists
    output_file = Path(tmp_path) / "ILLN-01.md"
    assert output_file.exists()

    # Check file content
    content = output_file.read_text()

    assert "# ILLN-01" in content
    assert "## Calibration" in content
    assert "| Values" in content
    assert "| Short Description" in content
    assert "1.0.0" in content
    assert "| laser events |" in content


def test_get_calibration_data(db_config, tmp_path):
    args = {
        "model_version": "6.0.0",
    }
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)

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


def test_get_array_element_parameter_data_simple(tmp_test_directory, monkeypatch):
    """Simple test for get_array_element_parameter_data using mocked DB calls."""

    # Prepare ReadParameters with a harmless config and an output path
    args = {"telescope": "LSTN-01", "site": "North", "model_version": "1.0.0", "observatory": None}
    rp = ReadParameters(db_config=None, args=args, output_path=tmp_test_directory)

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

    # Test that instrument-specific parameters are wrapped in bold/italic markers."""
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


def test_get_array_element_parameter_data_file_parameter(tmp_test_directory, monkeypatch):
    """Test that file parameters are converted to markdown links using _convert_to_md."""

    args = {"telescope": "LSTN-01", "site": "North", "model_version": "1.0.0", "observatory": None}
    rp = ReadParameters(db_config=None, args=args, output_path=tmp_test_directory)
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
    def _fake_convert_to_md(self, parameter, parameter_version, input_file):
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


def test_plot_camera_config(tmp_test_directory, db_config, mocker):
    args = {"telescope": "LSTN-01", "site": "North", "model_version": "6.0.0"}
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_test_directory)

    # Mock input file and output path
    input_file = Path(tmp_test_directory / "camera_config_file.dat")
    input_file.touch()
    plot_name = input_file.stem.replace(".", "-")
    plot_path = Path(tmp_test_directory / f"{plot_name}.png")

    # Mock plot_pixels.plot to avoid actual plotting
    mock_plot = mocker.patch("simtools.visualization.plot_pixels.plot")

    # Test when plot does not exist
    result = read_parameters._plot_camera_config(
        "camera_config_file", "1.0.0", input_file, tmp_test_directory
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
        "camera_config_file", "1.0.0", input_file, tmp_test_directory
    )
    assert result == [plot_name]
    mock_plot.assert_not_called()


def test_is_markdown_link():
    read_parameters = ReadParameters(db_config=None, args={}, output_path=Path())

    assert read_parameters.is_markdown_link("[example](http://example.com)") is True
    assert read_parameters.is_markdown_link("[text](target)") is True
    assert read_parameters.is_markdown_link("not a link") is False
    assert read_parameters.is_markdown_link("[missing target]") is False
    assert read_parameters.is_markdown_link("(missing text)") is False
    assert read_parameters.is_markdown_link("[text](target") is False
    assert read_parameters.is_markdown_link("[text]target)") is False
    assert read_parameters.is_markdown_link("") is False


def test_write_file_flag_section_non_camera(db_config, tmp_path):
    """Parameter not 'camera_config_file' should write a parameter plot link."""
    rp = ReadParameters(
        db_config=db_config,
        args={"telescope": "LSTN-01", "site": "North", "model_version": "6.0.0"},
        output_path=tmp_path,
    )

    # Prepare comparison_data for a non-camera parameter
    comparison_data = {
        "param_x": [{"parameter_version": "1.0.0", "model_version": "6.0.0", "file_flag": True}]
    }

    # Monkeypatch telescope identifier to a simple string to avoid DB calls
    rp._get_telescope_identifier = lambda mv=None: "LSTN-01"

    buf = StringIO()
    with patch("simtools.io.io_handler.IOHandler.get_output_directory", return_value=tmp_path):
        rp._write_file_flag_section(buf, "param_x", comparison_data)
    content = buf.getvalue()

    assert "The latest parameter version is plotted below." in content
    assert "![Parameter plot.]" in content


def test_write_file_flag_section_camera_config_file_with_match(tmp_path, db_config):
    """camera_config_file branch should extract filename from markdown link and write camera plot link."""
    rp = ReadParameters(
        db_config=db_config,
        args={"telescope": "LSTN-01", "site": "North", "model_version": "6.0.0"},
        output_path=tmp_path,
    )

    # Latest entry contains a markdown link with filename
    comparison_data = {
        "camera_config_file": [
            {
                "parameter_version": "1.0.0",
                "model_version": "6.0.0",
                "file_flag": True,
                "value": "[cam_config.dat](link)",
            }
        ]
    }

    # Ensure _get_telescope_identifier doesn't error
    rp._get_telescope_identifier = lambda mv=None: "LSTN-01"

    buf = StringIO()
    with patch("simtools.io.io_handler.IOHandler.get_output_directory", return_value=tmp_path):
        rp._write_file_flag_section(buf, "camera_config_file", comparison_data)
    content = buf.getvalue()

    assert "The latest parameter version is plotted below." in content
    assert "![Camera configuration plot.]" in content
    assert ".png" in content


def test_write_single_calibration_report_emphasis(tmp_path, db_config):
    """Ensure emphasized parameters keep emphasis and underscores are replaced by spaces in display."""
    output_file = tmp_path / "ILLN-01.md"

    rp = ReadParameters(db_config=db_config, args={"model_version": "6.0.0"}, output_path=tmp_path)

    # Build data grouped by class with one emphasized and one normal parameter
    data = [
        ["Calibration", "***laser_events***", "1.0.0", "10", "Desc", None],
        ["Calibration", "pedestal_events", "1.0.0", "100", "Desc2", "Short2"],
    ]

    # Call the writer directly; patch names.is_design_type to avoid validating the synthetic name
    with patch.object(names, "is_design_type", return_value=False):
        rp._write_single_calibration_report(output_file, "ILLN-01", data, "ILLN-design")

    content = output_file.read_text(encoding="utf-8")
    # Emphasized name should preserve bold/italics and replace underscore with space
    assert "***laser events***" in content
    # Normal parameter should have spaces for readability
    assert "pedestal events" in content


def test_write_file_flag_section_camera_config_latest_value_none(tmp_path, db_config):
    """camera_config_file branch: latest value for latest model exists but is None."""

    rp = ReadParameters(
        db_config=db_config,
        args={"telescope": "LSTN-01", "site": "North", "model_version": "6.0.0"},
        output_path=tmp_path,
    )

    # Create comparison_data where the latest model version is present but its value is None
    comparison_data = {
        "camera_config_file": [
            {
                "parameter_version": "1.0.0",
                "model_version": "6.0.0",
                "file_flag": True,
                "value": None,
            }
        ]
    }

    rp._get_telescope_identifier = lambda mv=None: "LSTN-01"

    buf = StringIO()
    rp._write_file_flag_section(buf, "camera_config_file", comparison_data)
    content = buf.getvalue()

    # The method writes the intro line and then early-returns when latest_value is None
    assert "The latest parameter version is plotted below." in content
    # No image link should be present
    assert "![Camera configuration plot.]" not in content


def test_write_file_flag_section_camera_config_no_markdown_match(tmp_path, db_config):
    """camera_config_file branch: latest value exists but does not contain a markdown link."""
    rp = ReadParameters(
        db_config=db_config,
        args={"telescope": "LSTN-01", "site": "North", "model_version": "6.0.0"},
        output_path=tmp_path,
    )

    # Latest entry contains a plain string without a markdown link
    comparison_data = {
        "camera_config_file": [
            {
                "parameter_version": "1.0.0",
                "model_version": "6.0.0",
                "file_flag": True,
                "value": "no link here",
            }
        ]
    }

    rp._get_telescope_identifier = lambda mv=None: "LSTN-01"

    buf = StringIO()
    rp._write_file_flag_section(buf, "camera_config_file", comparison_data)
    content = buf.getvalue()

    # Intro present, but regex didn't match so no camera image link should be written
    assert "The latest parameter version is plotted below." in content
    assert "![Camera configuration plot.]" not in content


def test_write_single_calibration_report_non_str_param(tmp_path, db_config):
    """Ensure non-string parameter names are written unchanged."""
    output_file = tmp_path / "ILLN-02.md"

    rp = ReadParameters(db_config=db_config, args={"model_version": "6.0.0"}, output_path=tmp_path)

    # Parameter value is an int (non-str) to hit the 'else: new_param = param' branch
    data = [["Calibration", 12345, "1.0.0", "10", "Desc", None]]

    rp._write_single_calibration_report(output_file, "ILLN-02", data, "ILLN-design")

    content = output_file.read_text(encoding="utf-8")
    # The numeric parameter name should appear unchanged in the output
    assert "12345" in content


def test_generate_model_parameter_reports_for_devices(db_config, tmp_path):
    """Test generating model parameter reports for calibration devices."""
    args = {"all_sites": True, "all_telescopes": True}
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=tmp_path)

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
