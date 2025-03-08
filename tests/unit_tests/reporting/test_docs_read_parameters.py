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


def test_produce_model_parameter_reports(telescope_model_lst, io_handler, db_config, mocker):
    output_path = io_handler.get_output_directory(
        label="reports", sub_dir=f"parameters/{telescope_model_lst.name}"
    )
    read_parameters = ReadParameters(
        db_config=db_config, telescope_model=telescope_model_lst, output_path=output_path
    )

    mock_params = {
        "focal_length": {"instrument": telescope_model_lst.name, "value": "2800.0", "unit": "cm"}
    }
    mocker.patch.object(telescope_model_lst.db, "get_model_parameters", return_value=mock_params)
    mock_comparison_data = [
        {
            "model_version": "5.0.0",
            "parameter_version": "1.0.0",
            "value": "2800.0 cm",
            "description": "Nominal overall focal length of the entire telescope.",
        },
        {
            "model_version": "4.0.0",
            "parameter_version": "1.0.0",
            "value": "2800.0 cm",
            "description": "Nominal overall focal length of the entire telescope.",
        },
    ]
    mocker.patch.object(
        read_parameters, "_compare_parameter_across_versions", return_value=mock_comparison_data
    )

    read_parameters.produce_model_parameter_reports()

    # Verify file was created with correct content
    file_path = output_path / "focal_length.md"
    assert file_path.exists()

    with open(file_path) as f:
        content = f.read()
        assert "# focal_length" in content
        assert f"**Telescope**: {telescope_model_lst.name}" in content
        assert "| Model Version      | Parameter Version      | Value                |" in content
        assert "| 5.0.0 | 1.0.0 |2800.0 cm |" in content
        assert "| 4.0.0 | 1.0.0 |2800.0 cm |" in content


def test__compare_parameter_across_versions_multiple_versions(db_config, mocker):
    """Test comparing parameters across multiple versions"""
    mock_telescope_model = mocker.Mock()
    mock_telescope_model.site = "South"
    mock_telescope_model.name = "MSTS-12"
    mock_output_path = mocker.Mock()

    read_params = ReadParameters(db_config, mock_telescope_model, mock_output_path)

    mock_telescope_model.db.get_model_versions.return_value = ["5.0.0", "6.0.0"]

    mock_telescope_instance = mocker.Mock()
    mock_telescope_instance.has_parameter.return_value = True
    mocker.patch(
        "simtools.model.telescope_model.TelescopeModel", return_value=mock_telescope_instance
    )

    read_params.get_array_element_parameter_data = mocker.Mock()
    read_params.get_array_element_parameter_data.side_effect = [
        [["Structure", "focal_length", "1.0.0", "100 cm", "Test description", "Short desc"]],
        [["Structure", "focal_length", "1.0.0", "90 cm", "Test description", "Short desc"]],
    ]

    result = read_params._compare_parameter_across_versions("focal_length")
    expected_values = [
        {"model_version": "5.0.0", "value": "90 cm"},
        {"model_version": "6.0.0", "value": "100 cm"},
    ]

    assert len(result) == 2
    for expected in expected_values:
        assert any(
            entry["model_version"] == expected["model_version"]
            and entry["value"] == expected["value"]
            for entry in result
        )


def test__compare_parameter_across_versions_parameter_not_found(db_config, mocker):
    """Test comparing parameters when parameter not found in model"""
    mock_telescope_model = mocker.Mock()
    mock_telescope_model.site = "North"
    mock_telescope_model.name = "LSTN-04"
    mock_output_path = mocker.Mock()

    read_params = ReadParameters(db_config, mock_telescope_model, mock_output_path)

    mock_telescope_model.db.get_model_versions.return_value = ["5.0.0", "6.0.0"]
    mock_telescope_instance = mocker.Mock()
    mock_telescope_instance.has_parameter.return_value = False
    mocker.patch(
        "simtools.model.telescope_model.TelescopeModel", return_value=mock_telescope_instance
    )

    result = read_params._compare_parameter_across_versions("nonexistent_param")

    assert result == []
