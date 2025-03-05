import astropy.units as u
import pytest

from simtools.reporting.docs_read_parameters import ReadParameters


def test_get_all_parameter_descriptions(telescope_model_lst, io_handler, db_config):
    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(
        db_config=db_config, telescope_model=telescope_model_lst, output_path=output_path
    )

    # Call get_all_parameter_descriptions
    descriptions, short_descriptions, inst_class = read_parameters.get_all_parameter_descriptions()

    assert isinstance(descriptions.get("focal_length"), str)
    assert isinstance(short_descriptions.get("focal_length"), str)
    assert isinstance(inst_class.get("focal_length"), str)


def test_get_telescope_parameter_data(telescope_model_lst, io_handler, db_config):
    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(
        db_config=db_config, telescope_model=telescope_model_lst, output_path=output_path
    )

    result = read_parameters.get_array_element_parameter_data(telescope_model_lst)

    # Assert the result contains the expected data
    if result[1] == "focal_length":
        assert result[0] == "Structure"
        assert result[3] == (2800.0 * u.cm)
        assert result[4] == "Nominal overall focal length of the entire telescope."


def test_produce_array_element_report(telescope_model_lst, io_handler, db_config):
    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(
        db_config=db_config, telescope_model=telescope_model_lst, output_path=output_path
    )

    read_parameters.produce_array_element_report()

    file_path = output_path / f"{telescope_model_lst.name}.md"
    assert file_path.exists()


def test_produce_model_parameter_reports(telescope_model_lst, io_handler, db_config):
    output_path = io_handler.get_output_directory(
        label="reports", sub_dir=f"parameters/{telescope_model_lst.name}"
    )
    read_parameters = ReadParameters(
        db_config=db_config, telescope_model=telescope_model_lst, output_path=output_path
    )

    read_parameters.produce_model_parameter_reports()

    file_path = output_path / "quantum_efficiency.md"
    assert file_path.exists()


def test__convert_to_md(telescope_model_lst, io_handler, db_config):
    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(
        db_config=db_config, telescope_model=telescope_model_lst, output_path=output_path
    )

    # testing with invalid file
    with pytest.raises(FileNotFoundError, match="Data file not found: "):
        read_parameters._convert_to_md("invalid-file.dat")

    # testing with valid file
    new_file = read_parameters._convert_to_md("tests/resources/spe_LST_2022-04-27_AP2.0e-4.dat")
    assert isinstance(new_file, str)


def test__compare_parameter_across_versions(telescope_model_lst, io_handler, db_config):
    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(
        db_config=db_config, telescope_model=telescope_model_lst, output_path=output_path
    )

    qe_comparison = read_parameters._compare_parameter_across_versions("quantum_efficiency")
    assert qe_comparison[0]["model_version"] != qe_comparison[1]["model_version"]
    assert qe_comparison["model_version" == "5.0.0"]["parameter_version"] == "1.0.0"
