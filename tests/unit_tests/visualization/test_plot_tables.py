#!/usr/bin/python3

from unittest import mock

import pytest

from simtools.visualization import plot_tables


@mock.patch("simtools.visualization.plot_tables.visualize")
@mock.patch("simtools.visualization.plot_tables.read_table_data")
def test_plot(mock_read_table_data, mock_visualize):
    config = {
        "tables": [
            {
                "label": "test_table",
                "parameter": "test_param",
                "site": "test_site",
                "column_x": "x",
                "column_y": "y",
            }
        ]
    }
    output_file = "output.png"
    mock_data = {"test_table": "mock_data"}
    mock_config = {"xtitle": "X", "ytitle": "Y"}
    mock_read_table_data.return_value = (mock_data, mock_config)
    mock_fig = mock.MagicMock()
    mock_visualize.plot_1d.return_value = mock_fig

    plot_tables.plot(config, output_file)

    mock_read_table_data.assert_called_once_with(config, None)
    # Check that plot_1d was called with merged configuration
    expected_config = plot_tables._get_default_plot_options()
    expected_config.update(config)
    expected_config.update(mock_config)
    mock_visualize.plot_1d.assert_called_once_with(mock_data, **expected_config)
    mock_visualize.save_figure.assert_called_once_with(mock_fig, output_file)


@mock.patch("simtools.visualization.plot_tables.gen.get_structure_array_from_table")
@mock.patch("simtools.visualization.plot_tables.db_handler.DatabaseHandler")
def test_read_astropy_table_data_from_file(mock_db_handler_class, mock_get_structure_array):
    config = {
        "tables": [
            {
                "label": "test_table",
                "parameter": "mirror_reflectivity",  # Use a known parameter type
                "file_name": "test_file",
                "type": "ascii.ecsv",
                "column_x": "wavelength",
                "column_y": "reflectivity",
                "site": "test_site",
            },
        ]
    }

    # Setup mock database handler
    mock_db_handler = mock_db_handler_class.return_value
    mock_table = mock.MagicMock()
    mock_table.colnames = ["wavelength", "reflectivity"]
    mock_db_handler.export_model_file.return_value = mock_table

    # Setup mock structure array
    mock_get_structure_array.return_value = mock.MagicMock()

    data, config_updates = plot_tables.read_table_data(config, None)

    # Verify the result
    assert "test_table" in data
    assert isinstance(config_updates, dict)
    mock_db_handler.export_model_file.assert_called_once_with(
        parameter="mirror_reflectivity",
        site="test_site",
        array_element_name=None,
        parameter_version=None,
        model_version=None,
        export_file_as_table=True,
    )


@mock.patch("simtools.visualization.plot_tables.gen.get_structure_array_from_table")
@mock.patch("simtools.visualization.plot_tables.db_handler.DatabaseHandler")
def test_read_table_data_from_model_database(mock_db_handler_class, mock_get_structure_array):
    config = {
        "tables": [
            {
                "label": "test_table",
                "parameter": "test_parameter",
                "site": "test_site",
                "telescope": "test_telescope",
                "model_version": "test_version",
                "column_x": "x",
                "column_y": "y",
            }
        ]
    }
    db_config = None

    # Setup mock database handler
    mock_db_handler = mock_db_handler_class.return_value
    mock_table = mock.MagicMock()
    mock_table.colnames = ["x", "y"]
    mock_db_handler.export_model_file.return_value = mock_table

    # Setup mock structure array
    mock_structure_array = mock.MagicMock()
    mock_get_structure_array.return_value = mock_structure_array

    data, config_updates = plot_tables.read_table_data(config, db_config)

    # Verify database interactions
    mock_db_handler_class.assert_called_once_with(mongo_db_config=db_config)
    mock_db_handler.export_model_file.assert_called_once_with(
        parameter="test_parameter",
        site="test_site",
        array_element_name="test_telescope",
        parameter_version=None,
        model_version="test_version",
        export_file_as_table=True,
    )

    # Verify data conversion
    mock_get_structure_array.assert_called_once_with(mock_table, ["x", "y", None, None])
    assert data == {"test_table": mock_structure_array}


def test_read_table_data_no_table_data_defined():
    config = {
        "tables": [
            {
                "label": "test_table",
                "column_x": "x",
                "column_y": "y",
                # Missing required parameter and site fields
            }
        ]
    }
    db_config = None

    with pytest.raises(KeyError):
        plot_tables.read_table_data(config, db_config)


@mock.patch("simtools.visualization.plot_tables.gen.get_structure_array_from_table")
@mock.patch("simtools.visualization.plot_tables.db_handler.DatabaseHandler")
def test_export_model_file(mock_db_handler_class, mock_get_structure_array):
    table_config = {
        "site": "test_site",
        "telescope": "test_telescope",
        "model_version": "test_version",
        "parameter": "test_parameter",
        "label": "test_label",
        "column_x": "x",
        "column_y": "y",
    }
    db_config = None

    # Set up mocks
    mock_db_handler = mock_db_handler_class.return_value
    mock_table = mock.MagicMock()
    mock_table.colnames = ["x", "y"]
    mock_db_handler.export_model_file.return_value = mock_table
    mock_get_structure_array.return_value = mock.MagicMock()

    config = {"tables": [table_config]}
    data, _ = plot_tables.read_table_data(config, db_config)

    # Verify database interaction
    mock_db_handler_class.assert_called_once_with(mongo_db_config=db_config)
    mock_db_handler.export_model_file.assert_called_once_with(
        parameter="test_parameter",
        site="test_site",
        array_element_name="test_telescope",
        parameter_version=None,
        model_version="test_version",
        export_file_as_table=True,
    )
    assert "test_label" in data


@mock.patch("simtools.visualization.plot_tables.gen.get_structure_array_from_table")
@mock.patch("simtools.visualization.plot_tables.db_handler.DatabaseHandler")
def test_export_model_file_site_only(mock_db_handler_class, mock_get_structure_array):
    table_config = {
        "site": "test_site",
        "model_version": "test_version",
        "parameter": "test_parameter",
        "label": "test_label",
        "column_x": "x",
        "column_y": "y",
    }
    db_config = None

    # Set up mocks
    mock_db_handler = mock_db_handler_class.return_value
    mock_table = mock.MagicMock()
    mock_table.colnames = ["x", "y"]
    mock_db_handler.export_model_file.return_value = mock_table
    mock_get_structure_array.return_value = mock.MagicMock()

    config = {"tables": [table_config]}
    data, _ = plot_tables.read_table_data(config, db_config)

    # Verify database interaction
    mock_db_handler_class.assert_called_once_with(mongo_db_config=db_config)
    mock_db_handler.export_model_file.assert_called_once_with(
        parameter="test_parameter",
        site="test_site",
        array_element_name=None,  # No telescope specified
        parameter_version=None,
        model_version="test_version",
        export_file_as_table=True,
    )
    assert "test_label" in data


@mock.patch("simtools.visualization.plot_tables.db_handler.DatabaseHandler")
def test_read_table_and_normalize(mock_db_handler_class):
    import numpy as np
    from astropy.table import Table

    # Create a mock table with test data
    test_data = Table(
        {
            "amplitude": [1.0, 2.0, 3.0],
            "response": [2.0, 4.0, 6.0],  # Values that will be normalized
        }
    )
    mock_db = mock_db_handler_class.return_value
    mock_db.export_model_file.return_value = test_data

    config = {
        "tables": [
            {
                "parameter": "test_param",
                "site": "test_site",
                "label": "test_table",
                "column_x": "amplitude",
                "column_y": "response",
                "normalize_y": True,
            }
        ]
    }

    with mock.patch(
        "simtools.visualization.plot_tables.gen.get_structure_array_from_table"
    ) as mock_array:
        # Mock the structure array to return normalized data
        mock_array.return_value = np.array(
            [(1.0, 0.333), (2.0, 0.667), (3.0, 1.0)],
            dtype=[("amplitude", "<f8"), ("response", "<f8")],
        )

        data, _ = plot_tables.read_table_data(config, None)
        assert isinstance(data, dict)
        assert "test_table" in data
        assert np.isclose(data["test_table"]["response"].max(), 1.0)


def test_get_default_plot_options():
    """Test default plot options."""
    options = plot_tables._get_default_plot_options()

    # Test all default values
    assert options["xscale"] == "linear"
    assert options["yscale"] == "linear"
    assert options["xlim"] is None
    assert options["ylim"] is None
    assert options["title"] is None
    assert options["palette"] == "default"
    assert options["no_legend"] is False
    assert options["big_plot"] is False
    assert options["no_markers"] is False
    assert options["empty_markers"] is False
    assert options["plot_ratio"] is False
    assert options["plot_difference"] is False
    assert options["marker"] is None
    assert options["linestyle"] == "-"


def test_update_config_from_columns():
    """Test configuration update from column information."""
    # Test with no columns
    assert plot_tables._update_config_from_columns(None) == {}
    assert plot_tables._update_config_from_columns([]) == {}

    # Test with columns without units
    columns = [
        {"description": "X Description", "name": "x"},
        {"description": "Y Description", "name": "y"},
    ]
    config = plot_tables._update_config_from_columns(columns)
    assert config["xtitle"] == "X Description"
    assert config["ytitle"] == "Y Description"

    # Test with columns with units
    columns = [
        {"description": "X Description", "name": "x", "unit": "nm"},
        {"description": "Y Description", "name": "y", "unit": "deg"},
    ]
    config = plot_tables._update_config_from_columns(columns)
    assert config["xtitle"] == "X Description [nm]"
    assert config["ytitle"] == "Y Description [deg]"


def test_infer_parameter_type():
    """Test parameter type inference from filenames."""
    # Test photoelectron spectrum files
    assert plot_tables._infer_parameter_type("spe_test.dat") == "pm_photoelectron_spectrum"
    assert plot_tables._infer_parameter_type("SPE_file.txt") == "pm_photoelectron_spectrum"

    # Test pulse shape files
    assert plot_tables._infer_parameter_type("pulse_shape.dat") == "pulse_shape"
    assert plot_tables._infer_parameter_type("PULSE_test.txt") == "pulse_shape"

    # Test quantum efficiency files
    assert plot_tables._infer_parameter_type("qe_data.dat") == "quantum_efficiency"
    assert plot_tables._infer_parameter_type("QE_test.txt") == "quantum_efficiency"

    # Test reflectivity files
    assert plot_tables._infer_parameter_type("ref_mirror.dat") == "mirror_reflectivity"
    assert plot_tables._infer_parameter_type("REF_test.txt") == "mirror_reflectivity"

    # Test unknown file types
    assert plot_tables._infer_parameter_type("unknown.dat") is None
    assert plot_tables._infer_parameter_type("test.txt") is None


def test_handle_reflectivity_data():
    """Test handling of multi-angle reflectivity data."""
    from astropy.table import Table

    # Create test table with multi-angle reflectivity data
    wavelengths = [300, 400, 500]
    refl_0 = [0.9, 0.91, 0.92]
    refl_10 = [0.89, 0.90, 0.91]
    refl_20 = [0.88, 0.89, 0.90]

    test_table = Table(
        {
            "wavelength": wavelengths,
            "reflectivity_0deg": refl_0,
            "reflectivity_10deg": refl_10,
            "reflectivity_20deg": refl_20,
        }
    )

    config = {"label": "Test Mirror"}
    result = plot_tables._handle_reflectivity_data(test_table, config)

    # Check all angles are processed
    assert "Test Mirror 0°" in result
    assert "Test Mirror 10°" in result
    assert "Test Mirror 20°" in result

    # Check data structure
    for label, data in result.items():
        assert "wavelength" in data.dtype.names
        assert any("reflectivity" in name for name in data.dtype.names)
        assert len(data) == len(wavelengths)

    # Test with empty table
    empty_table = Table({"wavelength": [], "reflectivity_0deg": []})
    result = plot_tables._handle_reflectivity_data(empty_table, config)
    assert all(len(data) == 0 for data in result.values())


def test_select_values_from_table():
    """Test selection of values from table."""
    from astropy.table import Table

    # Create test table
    test_table = Table({"x": [1.0, 2.0, 3.0, 2.0, 4.0], "y": [10.0, 20.0, 30.0, 21.0, 40.0]})

    # Test exact match
    result = plot_tables._select_values_from_table(test_table, "x", 2.0)
    assert len(result) == 2
    assert all(x == 2.0 for x in result["x"])
    assert 20.0 in result["y"]
    assert 21.0 in result["y"]

    # Test with no matches
    result = plot_tables._select_values_from_table(test_table, "x", 5.0)
    assert len(result) == 0

    # Test with floating point comparison
    result = plot_tables._select_values_from_table(test_table, "x", 2.0000001)
    assert len(result) == 2  # Should match due to np.isclose

    # Test with single match
    result = plot_tables._select_values_from_table(test_table, "x", 4.0)
    assert len(result) == 1
    assert result["x"][0] == 4.0
    assert result["y"][0] == 40.0


def test_read_simtel_table():
    """Test reading of sim_telarray format tables."""
    import pytest

    # Mock the dependencies
    @mock.patch("simtools.visualization.plot_tables.simtel_table_reader")
    def test_with_mocks(mock_reader):
        # Set up mock data
        mock_columns = [
            {"name": "x", "description": "X Value", "unit": "nm"},
            {"name": "y", "description": "Y Value", "unit": None},
        ]
        mock_description = "Test Description"
        mock_data = ([[1.0, 2.0], [3.0, 4.0]], "metadata", 2, None)

        # Configure mocks
        mock_reader.get_column_definitions.return_value = (mock_columns, mock_description)
        mock_reader.read_data.return_value = mock_data

        # Test successful read
        result = plot_tables.read_simtel_table("test_param", "test_file.dat")

        assert result.meta["Name"] == "test_param"
        assert result.meta["Description"] == mock_description
        assert result.meta["File"] == "test_file.dat"
        assert len(result) == 2
        assert all(col in result.colnames for col in ["x", "y"])

        # Test error handling
        mock_reader.read_data.side_effect = Exception("Test error")
        with pytest.raises(Exception, match="Test error"):
            plot_tables.read_simtel_table("test_param", "test_file.dat")

    # Run the test
    test_with_mocks()


def test_get_column_definitions():
    """Test column definitions retrieval."""
    with mock.patch("simtools.visualization.plot_tables.simtel_table_reader") as mock_reader:
        # Test successful retrieval
        mock_reader.get_column_definitions.return_value = (
            [{"name": "x", "description": "X"}, {"name": "y", "description": "Y"}],
            "Test Description",
        )
        columns, desc = plot_tables._get_column_definitions("test_param", 2)
        assert len(columns) == 2
        assert desc == "Test Description"

        # Test error handling
        mock_reader.get_column_definitions.side_effect = ValueError("Test error")
        columns, desc = plot_tables._get_column_definitions("test_param", 2)
        assert columns is None
        assert desc is None

        # Test attribute error handling
        mock_reader.get_column_definitions.side_effect = AttributeError("Test error")
        columns, desc = plot_tables._get_column_definitions("test_param", 2)
        assert columns is None
        assert desc is None


def test_read_table_data_description_handling():
    """Test description and title handling in read_table_data."""
    config = {
        "tables": [
            {
                "parameter": "mirror_reflectivity",
                "site": "test_site",
                "label": "test_table",
                "column_x": "wavelength",
                "column_y": "reflectivity",
            }
        ]
    }

    with mock.patch("simtools.visualization.plot_tables.db_handler.DatabaseHandler") as mock_db:
        # Setup mock table
        mock_table = mock.MagicMock()
        mock_table.colnames = ["wavelength", "reflectivity"]
        mock_db.return_value.export_model_file.return_value = mock_table

        # Mock column definitions with description
        mock_columns = [
            {"name": "wavelength", "description": "Wavelength", "unit": "nm"},
            {"name": "reflectivity", "description": "Reflectivity", "unit": None},
        ]
        mock_description = "Test Description"

        with mock.patch(
            "simtools.visualization.plot_tables._get_column_definitions",
            return_value=(mock_columns, mock_description),
        ):
            with mock.patch(
                "simtools.visualization.plot_tables.gen.get_structure_array_from_table"
            ) as mock_array:
                mock_array.return_value = mock.MagicMock()

                data, config_updates = plot_tables.read_table_data(config, None)

                # Verify description was used for title
                assert config["tables"][0]["title"] == mock_description
                assert "xtitle" in config_updates
                assert "ytitle" in config_updates


def test_read_table_data_parameter_version_label():
    """Test parameter version label handling in read_table_data."""
    # Test with parameter_version (should become the label)
    config = {
        "tables": [
            {
                "parameter": "mirror_reflectivity",
                "parameter_version": "v1.0.0",  # Should be used as label
                "site": "test_site",
                "column_x": "wavelength",
                "column_y": "reflectivity",
            }
        ]
    }

    with mock.patch("simtools.visualization.plot_tables.db_handler.DatabaseHandler") as mock_db:
        # Setup mock table
        mock_table = mock.MagicMock()
        mock_table.colnames = ["wavelength", "reflectivity"]
        mock_db.return_value.export_model_file.return_value = mock_table

        with mock.patch(
            "simtools.visualization.plot_tables.gen.get_structure_array_from_table"
        ) as mock_array:
            mock_array.return_value = mock.MagicMock()
            data, _ = plot_tables.read_table_data(config, None)

            # Verify parameter_version was used as label (config is modified in-place)
            assert config["tables"][0]["label"] == "v1.0.0"
            assert "v1.0.0" in data

    # Test without parameter_version (should use parameter name as label)
    config = {
        "tables": [
            {
                "parameter": "mirror_reflectivity",
                "site": "test_site",
                "column_x": "wavelength",
                "column_y": "reflectivity",
            }
        ]
    }

    with mock.patch("simtools.visualization.plot_tables.db_handler.DatabaseHandler") as mock_db:
        mock_db.return_value.export_model_file.return_value = mock_table
        with mock.patch(
            "simtools.visualization.plot_tables.gen.get_structure_array_from_table"
        ) as mock_array:
            mock_array.return_value = mock.MagicMock()
            data, _ = plot_tables.read_table_data(config, None)

            # Verify parameter was used as label
            assert config["tables"][0]["label"] == "mirror_reflectivity"
            assert "mirror_reflectivity" in data


def test_prepare_table_data_error_cases():
    """Test error handling in _prepare_table_data."""
    from astropy.table import Table

    # Test missing column_x in config
    table = Table({"x": [1, 2], "y": [3, 4]})
    config = {
        "column_y": "y"  # Missing column_x
    }
    with pytest.raises(ValueError, match="Missing required column_x in configuration"):
        plot_tables._prepare_table_data(table, config)

    # Test missing column_y in config
    config = {
        "column_x": "x"  # Missing column_y
    }
    with pytest.raises(ValueError, match="Missing required column_y in configuration"):
        plot_tables._prepare_table_data(table, config)

    # Test column not found in table
    config = {"column_x": "x", "column_y": "nonexistent"}
    with pytest.raises(ValueError, match="Column 'nonexistent' not found in table"):
        plot_tables._prepare_table_data(table, config)
