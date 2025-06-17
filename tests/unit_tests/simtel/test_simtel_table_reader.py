#!/usr/bin/python3

import logging
from unittest import mock

import astropy.units as u
import pytest
from astropy.table import Table

from simtools.simtel import simtel_table_reader


@pytest.fixture
def spe_test_file():
    return "tests/resources/spe_LST_2022-04-27_AP2.0e-4.dat"


@pytest.fixture
def spe_meta_test_comment():
    return "Norm_spe processing of single-p.e. response."


def test_read_simtel_data(spe_test_file, spe_meta_test_comment):
    """Test reading of sim_telarray table file into strings."""
    data, meta, n_columns, n_dim = simtel_table_reader._read_simtel_data(spe_test_file)
    assert isinstance(meta, str)
    assert isinstance(data, list)
    assert len(data) > 0
    assert n_dim is None
    assert n_columns == 7


def test_read_simtel_data_empty_file():
    """Test reading empty file."""
    with mock.patch("pathlib.Path.open", mock.mock_open(read_data="")):
        data, meta, n_columns, n_dim = simtel_table_reader._read_simtel_data("empty.txt")
        assert len(data) == 0
        assert meta == ""
        assert n_columns == 0
        assert n_dim is None


def test_read_simtel_data_rpol():
    test_data = """#@RPOL@[ANGLE=] 2
    ANGLE=     0              10              20
    # Blabla
    200     0.9353535       0.9349322       0.9332631"""

    with mock.patch("pathlib.Path.open", mock.mock_open(read_data=test_data)) as mock_file:
        # Configure mock to split lines properly
        mock_file.return_value.readlines.return_value = test_data.splitlines()
        rows, meta, n_columns, n_dim_axis = simtel_table_reader._read_simtel_data("test_file")

        # Check basic metadata
        assert "Blabla" in meta
        assert n_columns == 4

        # Check data rows - there should be one row of data plus one row for angles
        assert len(rows) == 2
        # Check angle row format
        assert rows[0] == [None, 0.0, 10.0, 20.0]  # ANGLE= row
        # Check data row format
        assert rows[1] == [200.0, 0.9353535, 0.9349322, 0.9332631]


def test_read_simtel_table_to_table(spe_test_file, spe_meta_test_comment):
    """Test reading of sim_telarray pm_photoelectron_spectrum table file into astropy table."""
    parameter_name = "pm_photoelectron_spectrum"
    table = simtel_table_reader.read_simtel_table(parameter_name, spe_test_file)

    assert len(table) == 2101
    assert "amplitude" in table.columns
    assert "response" in table.columns
    assert "response_with_ap" in table.columns
    assert table["amplitude"].unit == "pe"
    assert table["response"].unit is None
    assert table["response_with_ap"].unit is None
    assert table.meta["Name"] == parameter_name
    assert table.meta["File"] == spe_test_file
    assert spe_meta_test_comment in table.meta["Context_from_sim_telarray"]


@mock.patch("pathlib.Path.open")
def test_read_simtel_table_invalid_parameter(mock_open):
    """Test reading with invalid parameter."""
    mock_open.return_value.__enter__.return_value.readlines.return_value = ["1.0 2.0 3.0"]
    with pytest.raises(
        ValueError, match="Unsupported parameter for sim_telarray table reading: not_a_parameter"
    ):
        simtel_table_reader.read_simtel_table("not_a_parameter", "test.txt")


def test_read_simtel_table_ecsv_format():
    """Test reading ECSV format file."""
    mock_table = Table([[1, 2], [3, 4]], names=["a", "b"])
    with mock.patch("astropy.table.Table.read", return_value=mock_table) as mock_read:
        result = simtel_table_reader.read_simtel_table("any_parameter", "test.ecsv")
        mock_read.assert_called_once_with("test.ecsv", format="ascii.ecsv")

        assert len(result) == len(mock_table)
        assert result.colnames == mock_table.colnames
        for col in result.colnames:
            assert (result[col] == mock_table[col]).all()


def test_data_columns_mirror_reflectivity():
    """Test mirror reflectivity column definitions."""
    # Test without angles
    cols, desc = simtel_table_reader._data_columns_mirror_reflectivity()
    assert len(cols) == 2
    assert cols[0]["name"] == "wavelength"
    assert cols[1]["name"] == "reflectivity"
    assert desc == "Mirror reflectivity"

    # Test with angles
    cols, desc = simtel_table_reader._data_columns_mirror_reflectivity(n_dim=[0, 10, 20])
    exp_angles = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    assert len(cols) == len(exp_angles) + 1  # wavelength + angle columns
    assert cols[0]["name"] == "wavelength"
    for i, angle in enumerate(exp_angles):
        assert cols[i + 1]["name"] == f"reflectivity_{angle}deg"


@mock.patch("simtools.utils.general.read_file_encoded_in_utf_or_latin")
def test_read_simtel_data_for_atmospheric_transmission(mock_read_file):
    """Test atmospheric transmission data reading."""
    test_data = """
    # H2= 2.156, H1= 2.206 2.306
    200 0.5 0.6
    201 0.7 0.8
    """
    mock_read_file.return_value = test_data.splitlines()

    table = simtel_table_reader._read_simtel_data_for_atmospheric_transmission("test.txt")
    assert isinstance(table, Table)
    assert "wavelength" in table.colnames
    assert isinstance(table.meta["observatory_level"], u.Quantity)
    assert table.meta["observatory_level"] == 2.156 * u.km


def test_read_simtel_data_for_atmospheric_transmission_errors(caplog):
    """Test error handling in atmospheric transmission reading."""
    with mock.patch("simtools.utils.general.read_file_encoded_in_utf_or_latin") as mock_read:
        # Test missing H1= header
        mock_read.return_value = ["# Invalid header", "200 0.5"]
        with pytest.raises(ValueError, match="Header with 'H1=' not found"):
            simtel_table_reader._read_simtel_data_for_atmospheric_transmission("test.txt")

        # Test malformed data line
        mock_read.return_value = ["# H2= 2.156, H1= 2.206 2.306", "invalid_line", "200 invalid"]
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            table = simtel_table_reader._read_simtel_data_for_atmospheric_transmission("test.txt")
            assert len(table) == 0  # No valid data should be processed
            assert "Skipping malformed line" in caplog.text


def test_get_column_definitions():
    """Test column definitions for various parameters."""
    # Test ray-tracing parameter
    cols, desc = simtel_table_reader.get_column_definitions("ray-tracing")
    assert len(cols) == 5
    assert all(isinstance(col, dict) for col in cols)

    # Test mirror_reflectivity parameter
    cols, desc = simtel_table_reader.get_column_definitions("mirror_reflectivity", n_columns=2)
    assert len(cols) == 2
    assert cols[0]["name"] == "wavelength"

    # Test invalid parameter
    with pytest.raises(
        ValueError, match="Unsupported parameter for sim_telarray table reading: invalid_parameter"
    ):
        simtel_table_reader.get_column_definitions("invalid_parameter")


def test_read_simtel_data_value_conversion():
    """Test value conversion in _read_simtel_data."""
    test_data = """# Comment line
    1.0 %%% original: test # Line comment
    not_a_number 2.0 3.0"""
    with mock.patch("pathlib.Path.open", mock.mock_open(read_data=test_data)):
        data, _, _, _ = simtel_table_reader._read_simtel_data("test.txt")
        # Test row counts
        assert len(data) == 2
        # First row: check special markers are skipped
        assert data[0][0] == 1.0
        assert data[0][1] is None  # %%% is skipped
        # Second row: check invalid and valid numbers
        assert data[1][0] is None  # 'not_a_number' becomes None
        assert data[1][1] == 2.0
        assert data[1][2] == 3.0


def test_data_columns_pulse_shape():
    """Test pulse shape column definitions."""
    columns, _ = simtel_table_reader._data_columns_pulse_shape()
    assert len(columns) == 2
    assert columns[0]["name"] == "time"
    assert columns[0]["unit"] == "ns"
    assert columns[1]["name"] == "amplitude"
    assert columns[1]["unit"] is None


def test_data_columns_atmospheric_profile():
    """Test atmospheric profile column definitions."""
    columns, _ = simtel_table_reader._data_columns_atmospheric_profile()
    assert len(columns) == 7
    expected_units = ["km", "g/cm^3", "g/cm^2", None, "K", "mbar", None]
    for col, expected_unit in zip(columns, expected_units):
        assert col["unit"] == expected_unit


def test_data_columns_quantum_efficiency():
    """Test quantum efficiency column definitions."""
    columns, desc = simtel_table_reader._data_columns_quantum_efficiency()
    assert len(columns) == 3
    assert columns[0]["unit"] == "nm"
    assert columns[1]["unit"] is None
    assert columns[2]["unit"] is None
    assert desc == "Quantum efficiency vs wavelength"


def test_data_columns_camera_filter():
    """Test camera filter column definitions."""
    columns, desc = simtel_table_reader._data_columns_camera_filter()
    assert len(columns) == 2
    assert columns[0]["unit"] == "nm"
    assert columns[1]["unit"] is None
    assert desc == "Camera window transmission"


def test_data_columns_lightguide_efficiency_vs_wavelength():
    """Test lightguide efficiency vs wavelength column definitions."""
    columns, desc = simtel_table_reader._data_columns_lightguide_efficiency_vs_wavelength()
    assert len(columns) == 2
    assert columns[0]["unit"] == "nm"
    assert columns[1]["unit"] is None
    assert desc == "Light guide efficiency vs wavelength"


def test_data_columns_lightguide_efficiency_vs_incidence_angle():
    """Test lightguide efficiency vs incidence angle column definitions."""
    # Test with default columns
    columns, _ = simtel_table_reader._data_columns_lightguide_efficiency_vs_incidence_angle()
    assert len(columns) == 2
    assert columns[0]["unit"] == "deg"

    # Test with RMS column
    columns, _ = simtel_table_reader._data_columns_lightguide_efficiency_vs_incidence_angle(3)
    assert len(columns) == 3
    assert columns[2]["name"] == "efficiency_rms"


def test_data_columns_mirror_list():
    """Test mirror list column definitions."""
    columns, desc = simtel_table_reader._data_columns_mirror_list()
    assert len(columns) == 4
    assert all(col["unit"] == "m" for col in columns)
    assert desc == "Mirror positions and sizes"


def test_data_columns_secondary_mirror_reflectivity():
    """Test secondary mirror reflectivity column definitions."""
    # Test without dimensions
    columns, _ = simtel_table_reader._data_columns_secondary_mirror_reflectivity()
    assert len(columns) == 2
    assert columns[0]["name"] == "wavelength"
    assert columns[1]["name"] == "reflectivity"

    # Test with dimensions
    columns, desc = simtel_table_reader._data_columns_secondary_mirror_reflectivity(n_dim=[0, 10])
    assert len(columns) == 10  # wavelength + 9 angles (0-80 by 10)
    assert all("reflectivity_" in col["name"] for col in columns[1:])


def test_read_simtel_table_atmospheric_transmission(tmp_path):
    """Test reading atmospheric transmission data."""
    test_data = """# H2= 2.156, H1= 2.206 2.306 2.406
    200 0.5 0.6 0.7
    201 99999.0 0.8 0.9"""

    file_path = tmp_path / "atm.dat"
    file_path.write_text(test_data)

    with mock.patch(
        "simtools.utils.general.read_file_encoded_in_utf_or_latin",
        return_value=test_data.splitlines(),
    ):
        table = simtel_table_reader.read_simtel_table("atmospheric_transmission", str(file_path))
        assert "wavelength" in table.colnames
        assert "altitude" in table.colnames
        assert "extinction" in table.colnames
        assert len(table) == 5  # 2 wavelengths * (3 heights - 1 skipped value)


def test_adjust_columns_length():
    """Test column length adjustment function."""
    # Test truncating long rows
    assert simtel_table_reader._adjust_columns_length([[1.0, 2.0, 3.0]], 2) == [[1.0, 2.0]]

    # Test padding short rows
    assert simtel_table_reader._adjust_columns_length([[1.0]], 3) == [[1.0, 0.0, 0.0]]

    # Test empty input
    assert simtel_table_reader._adjust_columns_length([], 2) == []

    # Test mixed lengths
    rows = [[1.0, 2.0, 3.0], [1.0], [1.0, 2.0]]
    expected = [[1.0, 2.0], [1.0, 0.0], [1.0, 2.0]]
    assert simtel_table_reader._adjust_columns_length(rows, 2) == expected


def test_data_columns_fake_mirror_list():
    """Test fake mirror list column definitions."""
    columns, desc = simtel_table_reader._data_columns_fake_mirror_list()
    assert len(columns) == 4
    # Test all column properties
    expected_names = ["x_pos", "y_pos", "z_pos", "size"]
    expected_descriptions = ["X Position", "Y Position", "Z Position", "Mirror size"]
    for col, name, description in zip(columns, expected_names, expected_descriptions):
        assert col["unit"] == "m"
        assert col["name"] == name
        assert col["description"] == description
    assert desc == "Fake mirror positions and sizes"


def test_data_columns_mirror_segmentation():
    """Test mirror segmentation column definitions."""
    columns, desc = simtel_table_reader._data_columns_mirror_segmentation()
    assert len(columns) == 4
    units = [None, "m", "m", "m"]  # segment_id has no unit
    for col, unit in zip(columns, units):
        assert col["unit"] == unit
    assert desc == "Primary mirror segmentation layout"


def test_read_header_line_for_atmospheric_transmission():
    """Test reading header line for atmospheric transmission."""
    lines = ["# Some header", "# H2= 2.156, H1= 2.206 2.306 2.406", "# More header"]
    observatory_level, height_bins = (
        simtel_table_reader._read_header_line_for_atmospheric_transmission(lines, "test.txt")
    )
    assert observatory_level == 2.156 * u.km
    assert height_bins == [2.206, 2.306, 2.406]


def test_data_columns_nsb_reference_spectrum():
    """Test NSB reference spectrum column definitions."""
    columns, desc = simtel_table_reader._data_columns_nsb_reference_spectrum()
    assert len(columns) == 2
    assert columns[0]["unit"] == "nm"
    assert columns[1]["unit"] == "1.e9 / (nm s m^2 sr)"
    assert desc == "NSB reference spectrum"


def test_data_columns_ray_tracing():
    """Test ray tracing column definitions."""
    columns, desc = simtel_table_reader._data_columns_ray_tracing()
    assert len(columns) == 5
    expected_units = ["deg", "cm", "deg", "m2", "cm"]
    expected_names = ["Off-axis angle", "d80_cm", "d80_deg", "eff_area", "eff_flen"]
    for col, unit, name in zip(columns, expected_units, expected_names):
        assert col["unit"] == unit
        assert col["name"] == name
    assert desc == "Ray Tracing Data"


def test_data_columns_dsum_shaping():
    """Test digital sum shaping column definitions."""
    columns, desc = simtel_table_reader._data_columns_dsum_shaping()
    assert len(columns) == 2
    assert columns[0]["name"] == "time"
    assert columns[0]["unit"] == "ns"
    assert columns[1]["name"] == "amplitude"
    assert columns[1]["unit"] is None
    assert desc == "Digital sum shaping function"


def test_read_simtel_data_max_cols():
    """Test maximum column calculation in _read_simtel_data."""
    # Test with varying row lengths
    test_data = """# Comment line
    1.0 2.0 3.0
    4.0 5.0
    6.0 7.0 8.0 9.0"""
    with mock.patch("pathlib.Path.open", mock.mock_open(read_data=test_data)):
        data, meta, n_columns, n_dim = simtel_table_reader._read_simtel_data("test.txt")
        assert n_columns == 4  # Should be max length of any row
        assert len(data) == 3  # Should have 3 data rows

    # Test with only comments
    test_data = """# Comment 1
    # Comment 2"""
    with mock.patch("pathlib.Path.open", mock.mock_open(read_data=test_data)):
        data, meta, n_columns, n_dim = simtel_table_reader._read_simtel_data("test.txt")
        assert n_columns == 0  # No data rows means max_cols should be 0

    # Test with empty lines between data
    test_data = """1.0 2.0
        3.0 4.0 5.0
        6.0"""
    with mock.patch("pathlib.Path.open", mock.mock_open(read_data=test_data)):
        data, meta, n_columns, n_dim = simtel_table_reader._read_simtel_data("test.txt")
        assert n_columns == 3  # Should be max length despite empty lines


def test_read_simtel_table_edge_cases():
    """Test edge cases in read_simtel_table."""
    # Test dsum_shaping with empty data
    test_data = "# Empty data"
    with mock.patch("pathlib.Path.open", mock.mock_open(read_data=test_data)):
        table = simtel_table_reader.read_simtel_table("dsum_shaping", "test.txt")
        assert len(table) == 0
        assert "time" in table.colnames
        assert "amplitude" in table.colnames

    # Test regular parameter with empty data
    with mock.patch("pathlib.Path.open", mock.mock_open(read_data=test_data)):
        table = simtel_table_reader.read_simtel_table("mirror_list", "test.txt")
        assert len(table) == 0
        assert all(name in table.colnames for name in ["x_pos", "y_pos", "z_pos", "size"])


def test_read_simtel_data_for_atmospheric_transmission_table_creation():
    """Test table creation in atmospheric transmission data reading."""
    test_data = """# H2= 2.156, H1= 2.206
    # Invalid line that should be skipped
    200 0.5
    201 invalid_value
    202 99999.0"""

    with mock.patch(
        "simtools.utils.general.read_file_encoded_in_utf_or_latin",
        return_value=test_data.splitlines(),
    ):
        table = simtel_table_reader._read_simtel_data_for_atmospheric_transmission("test.txt")
        assert isinstance(table, Table)
        assert len(table) == 1  # Only one valid data point
        assert table["wavelength"][0] == 200
        assert table["extinction"][0] == 0.5
        assert "wavelength" in table.colnames
        assert "altitude" in table.colnames
        assert "extinction" in table.colnames
