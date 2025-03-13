from pathlib import Path

import astropy.units as u
import pytest

from simtools.reporting.docs_read_parameters import ReadParameters


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


def test_produce_array_element_report(telescope_model_lst, io_handler, db_config):
    args = {
        "telescope": telescope_model_lst.name,
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
    }
    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

    read_parameters.produce_array_element_report()

    file_path = output_path / f"{telescope_model_lst.name}.md"
    assert file_path.exists()


def test_produce_model_parameter_reports(io_handler, db_config):
    args = {"site": "North", "telescope": "LSTN-01"}
    output_path = io_handler.get_output_directory(
        label="reports", sub_dir=f"parameters/{args['telescope']}"
    )
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

    read_parameters.produce_model_parameter_reports()

    file_path = output_path / "quantum_efficiency.md"
    assert file_path.exists()


def test__convert_to_md(telescope_model_lst, io_handler, db_config):
    args = {
        "telescope": telescope_model_lst.name,
        "site": telescope_model_lst.site,
        "model_version": telescope_model_lst.model_version,
    }
    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

    # testing with invalid file
    with pytest.raises(FileNotFoundError, match="Data file not found: "):
        read_parameters._convert_to_md("invalid-file.dat")

    # testing with valid file
    new_file = read_parameters._convert_to_md("tests/resources/spe_LST_2022-04-27_AP2.0e-4.dat")
    assert isinstance(new_file, str)
    assert Path(output_path / new_file).exists()


def test__format_parameter_value(io_handler, db_config):
    output_path = io_handler.get_output_directory()
    read_parameters = ReadParameters(db_config=db_config, args={}, output_path=output_path)

    mock_data_1 = [[24.74, 9.0, 350.0, 1066.0], ["ns", "ns", "V", "V"], False]
    result_1 = read_parameters._format_parameter_value(*mock_data_1)
    assert result_1 == "24.74 ns, 9.0 ns, 350.0 V, 1066.0 V"

    mock_data_2 = [4.0, " ", None]
    result_2 = read_parameters._format_parameter_value(*mock_data_2)
    assert result_2 == "4.0"

    mock_data_3 = [
        [0.233591, 0.233591, 0.233591, 0.233591, 0.233591, 0.233591, 0.233591],
        "GHz",
        False,
    ]
    result_3 = read_parameters._format_parameter_value(*mock_data_3)
    assert result_3 == "all: 0.233591 GHz"

    mock_data_4 = [[1, 2, 3, 4], "m", None]
    result_4 = read_parameters._format_parameter_value(*mock_data_4)
    assert result_4 == "1 m, 2 m, 3 m, 4 m"


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
                "value": "qe_lst1_20200318_high+low.dat",
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
                "value": "qe_lst1_20200318_high+low.dat",
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
        },
    }

    comparison_data = read_parameters._compare_parameter_across_versions(
        mock_data, ["quantum_efficiency", "array_element_position_ground"]
    )
    qe_comparison = comparison_data.get("quantum_efficiency")
    assert qe_comparison["parameter_version" == "1.0.0"]["model_version"] == "6.0.0, 5.0.0"

    position_comparison = comparison_data.get("array_element_position_ground")
    assert position_comparison[0]["model_version"] != position_comparison[1]["model_version"]
    assert position_comparison["parameter_version" == "2.0.0"]["model_version"] == "6.0.0"
