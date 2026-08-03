import astropy.units as u
import h5py
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from simtools.io.table_handler import (
    _read_table_list_fits,
    _read_table_list_hdf5,
    _write_table_to_hdf5_file,
    group_table_rows,
    read_metadata_document,
    read_table_file_type,
    read_table_from_hdf5,
    read_table_list,
    read_tables,
    write_table_chunks,
    write_tables,
)

# Constants for repeated strings
TABLE_HANDLER_PATH = "simtools.io.table_handler"
READ_TABLE_FILE_TYPE = f"{TABLE_HANDLER_PATH}.read_table_file_type"
ASTROPY_TABLE_READ = "astropy.table.Table.read"
H5PY_FILE = "h5py.File"
TEST_TABLE_NAME = "test_table"

# Test file paths
TEST_FITS = "test.fits"
TEST_HDF5 = "test.hdf5"
TEST_H5 = "test.h5"
TEST_CSV = "test.csv"


@pytest.fixture
def mock_table():
    """Create a mock table with test data."""
    table = Table({"col1": [1, 2]})
    table.meta["EXTNAME"] = TEST_TABLE_NAME
    return table


@pytest.fixture
def mock_read_type(mocker):
    """Mock read_table_file_type."""
    return mocker.patch(READ_TABLE_FILE_TYPE)


@pytest.fixture
def mock_table_read(mocker):
    """Mock Table.read."""
    return mocker.patch(ASTROPY_TABLE_READ)


@pytest.fixture
def mock_table_write(mocker):
    """Mock Table.write."""
    return mocker.patch("astropy.table.Table.write")


@pytest.fixture
def mock_fits_objects(mocker):
    """Mock FITS objects and functions."""
    return {
        "primary_hdu": mocker.patch("astropy.io.fits.PrimaryHDU"),
        "table_hdu": mocker.patch("astropy.io.fits.BinTableHDU"),
        "hdul": mocker.patch("astropy.io.fits.HDUList"),
    }


@pytest.fixture
def mock_h5py_file(mocker):
    """Mock h5py.File context manager."""
    mock_file = mocker.MagicMock()
    mock_context = mocker.MagicMock()
    mock_context.__enter__.return_value = mock_file
    mocker.patch(H5PY_FILE, return_value=mock_context)
    return mock_file


def test_read_table_file_type_empty_list():
    with pytest.raises(ValueError, match=r"No input files provided."):
        read_table_file_type([])


def test_read_table_file_type_all_fits():
    input_files = [TEST_FITS, "test2.fits.gz"]
    assert read_table_file_type(input_files) == "FITS"


def test_read_table_file_type_all_hdf5():
    input_files = [TEST_HDF5, TEST_H5]
    assert read_table_file_type(input_files) == "HDF5"


def test_read_table_file_type_mixed_types():
    input_files = [TEST_FITS, TEST_HDF5]
    with pytest.raises(ValueError, match="All input files must be of the same type"):
        read_table_file_type(input_files)


def test_read_table_file_type_unsupported():
    input_files = ["test.txt"]
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_table_file_type(input_files)


def test_group_table_rows():
    table = Table(
        rows=[
            {"reference_id": "ref-1", "axis_index": 0, "value": 1},
            {"reference_id": "ref-1", "axis_index": 1, "value": 2},
            {"reference_id": "ref-2", "axis_index": 0, "value": 3},
        ]
    )

    grouped_by_reference = group_table_rows(table, "reference_id")
    grouped_by_reference_axis = group_table_rows(table, ["reference_id", "axis_index"])

    assert set(grouped_by_reference) == {"ref-1", "ref-2"}
    assert list(grouped_by_reference["ref-1"]["value"]) == [1, 2]
    assert set(grouped_by_reference_axis) == {("ref-1", 0), ("ref-1", 1), ("ref-2", 0)}
    assert grouped_by_reference_axis[("ref-1", 1)]["value"][0] == 2


def test_read_tables_hdf5(mocker):
    """Test reading tables from HDF5 file."""
    # Mock h5py.File context manager
    mock_h5file = mocker.MagicMock()
    mock_h5_context = mocker.MagicMock()
    mock_h5_context.__enter__.return_value = mock_h5file
    mocker.patch("h5py.File", return_value=mock_h5_context)

    mock_read = mocker.patch(ASTROPY_TABLE_READ)
    mock_table = Table({"col1": [1, 2]})
    mock_read.return_value = mock_table

    mock_file_type = mocker.patch(READ_TABLE_FILE_TYPE)
    mock_file_type.return_value = "HDF5"

    result = read_tables(TEST_H5, ["table1", "table2"])

    assert len(result) == 2
    assert all(name in result for name in ["table1", "table2"])
    assert mock_read.call_count == 2
    mock_read.assert_has_calls(
        [mocker.call(TEST_H5, path="table1"), mocker.call(TEST_H5, path="table2")]
    )


def test_read_tables_hdf5_with_selected_columns(mocker):
    """Test reading selected HDF5 table columns."""
    mock_reader = mocker.patch(f"{TABLE_HANDLER_PATH}.read_table_from_hdf5")
    mock_reader.return_value = Table({"col1": [1, 2]})

    result = read_tables(
        TEST_H5,
        ["table1", "table2"],
        file_type="HDF5",
        table_columns={"table1": ["col1"]},
    )

    assert len(result) == 2
    mock_reader.assert_has_calls(
        [
            mocker.call(TEST_H5, "table1", columns=["col1"]),
            mocker.call(TEST_H5, "table2", columns=None),
        ]
    )


def test_read_tables_unsupported_format(mocker):
    mock_file_type = mocker.patch(READ_TABLE_FILE_TYPE)
    mock_file_type.return_value = "CSV"

    with pytest.raises(ValueError, match="Unsupported file format"):
        read_tables(TEST_CSV, ["table1"])


def test_read_tables_explicit_file_type(mocker):
    mock_read = mocker.patch(ASTROPY_TABLE_READ)
    mock_table = Table({"col1": [1, 2]})
    mock_read.return_value = mock_table

    mock_file_type = mocker.patch(READ_TABLE_FILE_TYPE)

    result = read_tables(TEST_FITS, ["table1"], file_type="FITS")

    assert len(result) == 1
    mock_file_type.assert_not_called()
    mock_read.assert_called_once_with(TEST_FITS, hdu="table1")


def test_write_tables_fits(tmp_path, mock_table, mock_fits_objects):
    """Test writing tables in FITS format."""
    output_file = tmp_path / TEST_FITS

    write_tables([mock_table], output_file, file_type="FITS")

    mock_fits_objects["hdul"].assert_called_once()
    mock_fits_objects["hdul"].return_value.writeto.assert_called_once_with(
        output_file, checksum=False
    )


def test_write_tables_fits_overwrite_false(tmp_path, mock_table, mock_fits_objects, mocker):
    """Test writing tables in FITS format when overwrite is False and file exists."""
    output_file = tmp_path / TEST_FITS

    mocker.patch("pathlib.Path.exists", return_value=True)

    with pytest.raises(FileExistsError, match=r"^Output file "):
        write_tables([mock_table], output_file, overwrite_existing=False, file_type="FITS")


def test_write_tables_hdf5(tmp_path, mock_table):
    """Test writing tables in HDF5 format."""
    output_file = tmp_path / TEST_H5
    write_tables([mock_table], output_file, file_type="HDF5")

    with h5py.File(output_file, "r") as hdf5_file:
        assert hdf5_file.attrs["simtools_write_status"] == "complete"
        assert len(hdf5_file[TEST_TABLE_NAME]) == 2
        assert hdf5_file[TEST_TABLE_NAME].compression == "gzip"
        assert hdf5_file[TEST_TABLE_NAME].compression_opts == 6
    assert not list(tmp_path.glob(f"{TEST_H5}.incomplete-*"))


def test_write_tables_hdf5_unicode_string_columns(tmp_path):
    """Test that unicode string columns are converted to byte strings when writing to HDF5."""
    table = Table({"name": ["hello", "world"], "value": [1, 2]})
    table.meta["EXTNAME"] = TEST_TABLE_NAME
    output_file = tmp_path / TEST_H5

    write_tables([table], output_file, file_type="HDF5")

    with h5py.File(output_file, "r") as hdf5_file:
        dataset = hdf5_file[TEST_TABLE_NAME]
        assert dataset["name"].dtype.kind == "S"
        assert dataset["name"][:].tolist() == [b"hello", b"world"]


def test_write_tables_hdf5_object_dtype_string_columns(tmp_path):
    """Test that object-dtype columns with string values are serialized to HDF5."""
    col = np.array(["hello", "world"], dtype=object)
    table = Table({"name": col, "value": [1, 2]})
    table.meta["EXTNAME"] = TEST_TABLE_NAME
    output_file = tmp_path / TEST_H5

    write_tables([table], output_file, file_type="HDF5")

    with h5py.File(output_file, "r") as hdf5_file:
        dataset = hdf5_file[TEST_TABLE_NAME]
        assert dataset["name"].dtype.kind == "S"


def test_write_tables_hdf5_object_dtype_non_string_raises(tmp_path):
    """Test that object-dtype columns with non-string values raise TypeError when writing HDF5."""
    col = np.array(["hello", None], dtype=object)
    table = Table({"name": col, "value": [1, 2]})
    table.meta["EXTNAME"] = TEST_TABLE_NAME
    output_file = tmp_path / TEST_H5

    with pytest.raises(TypeError, match="non-string or missing values"):
        write_tables([table], output_file, file_type="HDF5")


def test_write_table_chunks_appends_and_widens_strings(tmp_path):
    """Append chunks without truncating strings that grow in later chunks."""
    first = Table({"name": ["a"], "value": [1]})
    second = Table({"name": ["a-longer-name"], "value": [2]})
    for table in (first, second):
        table.meta["EXTNAME"] = TEST_TABLE_NAME

    output_file = tmp_path / TEST_H5
    write_table_chunks([[first], [second]], output_file)

    with h5py.File(output_file) as hdf5_file:
        dataset = hdf5_file[TEST_TABLE_NAME]
        assert dataset["name"][:].tolist() == [b"a", b"a-longer-name"]
        assert dataset["value"][:].tolist() == [1, 2]
        assert dataset.dtype["name"].itemsize == len("a-longer-name")
        assert hdf5_file.attrs["simtools_write_status"] == "complete"


def test_write_table_chunks_failure_preserves_existing_output(tmp_path, mock_table):
    """Keep the published output unchanged when chunk generation fails."""
    output_file = tmp_path / TEST_H5
    output_file.write_bytes(b"existing output")

    def failing_chunks():
        yield [mock_table]
        raise RuntimeError("injected chunk failure")

    with pytest.raises(RuntimeError, match="injected chunk failure"):
        write_table_chunks(failing_chunks(), output_file)

    assert output_file.read_bytes() == b"existing output"
    incomplete_files = list(tmp_path.glob(f"{TEST_H5}.incomplete-*"))
    assert len(incomplete_files) == 1
    with h5py.File(incomplete_files[0]) as hdf5_file:
        assert hdf5_file.attrs["simtools_write_status"] == "incomplete"
        assert TEST_TABLE_NAME in hdf5_file


def test_write_tables_dict_input(tmp_path, mock_table):
    """Test writing dictionary of tables."""
    tables_dict = {"table1": mock_table}
    output_file = tmp_path / TEST_H5
    write_tables(tables_dict, output_file, file_type="HDF5")

    with h5py.File(output_file, "r") as hdf5_file:
        assert TEST_TABLE_NAME in hdf5_file


def test_write_and_read_named_metadata_documents(tmp_path, mock_table):
    """Store named JSON documents atomically alongside event tables."""
    output_file = tmp_path / TEST_H5
    write_tables(
        [mock_table],
        output_file,
        file_type="HDF5",
        metadata_documents={"METADATA": {"value": np.int64(3)}},
    )

    assert read_metadata_document(output_file, "METADATA") == {"value": 3}
    with h5py.File(output_file, "r") as hdf5_file:
        assert hdf5_file["METADATA"].shape == (1,)
        assert hdf5_file.attrs["simtools_write_status"] == "complete"


def test_write_tables_hdf5_failure_preserves_existing_output(tmp_path, mock_table, mocker):
    """A mid-write failure preserves prior output and leaves the partial file identifiable."""
    output_file = tmp_path / TEST_H5
    output_file.write_bytes(b"existing output")
    second_table = mock_table.copy()
    second_table.meta["EXTNAME"] = "second_table"
    original_writer = _write_table_to_hdf5_file
    write_count = 0

    def fail_after_first_table(table, hdf5_file, table_name):
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise RuntimeError("injected write failure")
        original_writer(table, hdf5_file, table_name)

    mocker.patch(
        f"{TABLE_HANDLER_PATH}._write_table_to_hdf5_file",
        side_effect=fail_after_first_table,
    )

    with pytest.raises(RuntimeError, match="injected write failure"):
        write_tables([mock_table, second_table], output_file, file_type="HDF5")

    assert output_file.read_bytes() == b"existing output"
    incomplete_files = list(tmp_path.glob(f"{TEST_H5}.incomplete-*"))
    assert len(incomplete_files) == 1
    with h5py.File(incomplete_files[0], "r") as hdf5_file:
        assert hdf5_file.attrs["simtools_write_status"] == "incomplete"
        assert TEST_TABLE_NAME in hdf5_file
        assert "second_table" not in hdf5_file


def test_write_tables_existing_file(tmp_path, mocker):
    """Test writing tables when output file exists."""
    mock_table = Table({"col1": [1, 2]})
    mock_table.meta["EXTNAME"] = TEST_TABLE_NAME

    output_file = tmp_path / TEST_FITS
    output_file.touch()  # Create the file

    mocker.patch("astropy.io.fits.PrimaryHDU")
    mocker.patch("astropy.io.fits.BinTableHDU")
    mock_hdul = mocker.patch("astropy.io.fits.HDUList")

    write_tables([mock_table], output_file, file_type="FITS")

    assert not output_file.exists()  # File should be deleted before writing
    mock_hdul.assert_called_once()


def test_write_tables_no_file_type(tmp_path, mock_table, mock_read_type, mock_fits_objects):
    """Test writing tables without explicit file type."""
    mock_read_type.return_value = "FITS"
    output_file = tmp_path / TEST_FITS

    write_tables([mock_table], output_file)

    mock_read_type.assert_called_once_with([output_file])
    mock_fits_objects["hdul"].assert_called_once()


def test_read_table_list_hdf5(mocker):
    """Test read_table_list with HDF5 file."""
    mock_read_type = mocker.patch(READ_TABLE_FILE_TYPE, return_value="HDF5")
    mock_read_hdf5 = mocker.patch(
        "simtools.io.table_handler._read_table_list_hdf5",
        return_value={"table1": ["table1"], "table2": ["table2"]},
    )

    result = read_table_list(TEST_H5, ["table1", "table2"])

    mock_read_type.assert_called_once_with(TEST_H5)
    mock_read_hdf5.assert_called_once_with(TEST_H5, ["table1", "table2"], False)
    assert result == {"table1": ["table1"], "table2": ["table2"]}


def test_read_table_list_fits(mocker):
    """Test read_table_list with FITS file."""
    mock_read_type = mocker.patch(READ_TABLE_FILE_TYPE, return_value="FITS")
    mock_read_fits = mocker.patch(
        "simtools.io.table_handler._read_table_list_fits",
        return_value={"table1": ["table1"], "table2": ["table2"]},
    )

    result = read_table_list(TEST_FITS, ["table1", "table2"], True)

    mock_read_type.assert_called_once_with(TEST_FITS)
    mock_read_fits.assert_called_once_with(TEST_FITS, ["table1", "table2"], True)
    assert result == {"table1": ["table1"], "table2": ["table2"]}


def test_read_table_list_unsupported_format(mocker):
    """Test read_table_list with unsupported file format."""
    mock_read_type = mocker.patch(READ_TABLE_FILE_TYPE, return_value="CSV")

    result = read_table_list(TEST_CSV, ["table1"])

    mock_read_type.assert_called_once_with(TEST_CSV)
    assert result is None


def test_read_table_list_hdf5_basic(mocker, mock_h5py_file):
    """Test reading basic HDF5 table list without indexed tables."""
    # Mock datasets
    dataset1 = mocker.MagicMock(spec=h5py.Dataset)
    dataset2 = mocker.MagicMock(spec=h5py.Dataset)
    mock_h5py_file.visititems.side_effect = lambda x: [
        x("table1", dataset1),
        x("table2", dataset2),
    ]

    result = _read_table_list_hdf5(TEST_H5, ["table1", "table2"], False)

    assert result == {"table1": ["table1"], "table2": ["table2"]}


def test_read_table_list_hdf5_with_indexed(mocker, mock_h5py_file):
    """Test reading HDF5 table list with indexed tables."""
    # Mock datasets
    datasets = {
        "table1": mocker.MagicMock(spec=h5py.Dataset),
        "table1_0": mocker.MagicMock(spec=h5py.Dataset),
        "table1_1": mocker.MagicMock(spec=h5py.Dataset),
        "table2": mocker.MagicMock(spec=h5py.Dataset),
        "table2_0": mocker.MagicMock(spec=h5py.Dataset),
    }

    def mock_visititems(visitor):
        for name, dataset in datasets.items():
            visitor(name, dataset)

    mock_h5py_file.visititems.side_effect = mock_visititems

    result = _read_table_list_hdf5(TEST_H5, ["table1", "table2"], True)

    assert result == {
        "table1": ["table1", "table1_0", "table1_1"],
        "table2": ["table2", "table2_0"],
    }


def test_read_table_list_hdf5_ignore_non_datasets(mocker, mock_h5py_file):
    """Test that non-dataset objects are ignored."""
    # Mock a group (not a dataset)
    group = mocker.MagicMock(spec=h5py.Group)
    dataset = mocker.MagicMock(spec=h5py.Dataset)

    def mock_visititems(visitor):
        visitor("table1", group)  # Should be ignored
        visitor("table1", dataset)  # Should be included

    mock_h5py_file.visititems.side_effect = mock_visititems

    result = _read_table_list_hdf5(TEST_H5, ["table1"], False)

    assert result == {"table1": ["table1"]}


def test_read_table_list_hdf5_ignore_invalid_suffix(mocker, mock_h5py_file):
    """Test that indexed tables with invalid suffixes are ignored."""
    dataset1 = mocker.MagicMock(spec=h5py.Dataset)
    dataset2 = mocker.MagicMock(spec=h5py.Dataset)

    def mock_visititems(visitor):
        visitor("table1", dataset1)
        visitor("table1_abc", dataset2)  # Invalid suffix

    mock_h5py_file.visititems.side_effect = mock_visititems

    result = _read_table_list_hdf5(TEST_H5, ["table1"], True)

    assert result == {"table1": ["table1"]}


def test_read_table_list_hdf5_empty_file(mock_h5py_file):
    """Test reading from an empty HDF5 file."""
    mock_h5py_file.visititems.side_effect = lambda x: None

    result = _read_table_list_hdf5(TEST_H5, ["table1", "table2"], False)

    assert result == {"table1": [], "table2": []}


def test_read_table_list_fits_basic(mocker):
    """Test reading basic FITS table list without indexed tables."""
    mock_primary = mocker.MagicMock(spec=fits.PrimaryHDU)
    mock_primary.name = "PRIMARY"

    mock_table1 = mocker.MagicMock(spec=fits.BinTableHDU)
    mock_table1.name = "table1"
    mock_table1.is_image = False

    mock_table2 = mocker.MagicMock(spec=fits.BinTableHDU)
    mock_table2.name = "table2"
    mock_table2.is_image = False

    mock_table3 = mocker.MagicMock(spec=fits.BinTableHDU)
    mock_table3.name = "table3"
    mock_table3.is_image = False

    mock_table4 = mocker.MagicMock(spec=fits.TableHDU)
    mock_table4.name = "table4"
    mock_table4.is_image = False

    mock_hdul = [mock_primary, mock_table1, mock_table2, mock_table3, mock_table4]

    # Create a context manager mock that returns our HDU list
    mock_fits_open = mocker.MagicMock()
    mock_fits_open.__enter__ = mocker.Mock(return_value=mock_hdul)
    mock_fits_open.__exit__ = mocker.Mock(return_value=None)

    mocker.patch("astropy.io.fits.open", return_value=mock_fits_open)

    result = _read_table_list_fits(TEST_FITS, ["table1", "table2"], False)

    assert result == {"table1": ["table1"], "table2": ["table2"]}


def test_read_table_list_fits_with_indexed(mocker):
    """Test reading FITS table list with indexed tables."""
    mock_primary = mocker.MagicMock(spec=fits.PrimaryHDU)
    mock_primary.name = "PRIMARY"

    mock_table1 = mocker.MagicMock(spec=fits.BinTableHDU)
    mock_table1.name = "table1"
    mock_table1.is_image = False

    mock_table1_0 = mocker.MagicMock(spec=fits.BinTableHDU)
    mock_table1_0.name = "table1_0"
    mock_table1_0.is_image = False

    mock_table1_1 = mocker.MagicMock(spec=fits.BinTableHDU)
    mock_table1_1.name = "table1_1"
    mock_table1_1.is_image = False

    mock_table2 = mocker.MagicMock(spec=fits.BinTableHDU)
    mock_table2.name = "table2"
    mock_table2.is_image = False

    mock_table2_0 = mocker.MagicMock(spec=fits.BinTableHDU)
    mock_table2_0.name = "table2_0"
    mock_table2_0.is_image = False

    mock_hdul = [
        mock_primary,
        mock_table1,
        mock_table1_0,
        mock_table1_1,
        mock_table2,
        mock_table2_0,
    ]

    # Create a context manager mock that returns our HDU list
    mock_fits_open = mocker.MagicMock()
    mock_fits_open.__enter__ = mocker.Mock(return_value=mock_hdul)
    mock_fits_open.__exit__ = mocker.Mock(return_value=None)

    mocker.patch("astropy.io.fits.open", return_value=mock_fits_open)

    result = _read_table_list_fits(TEST_FITS, ["table1", "table2"], True)

    assert result == {
        "table1": ["table1", "table1_0", "table1_1"],
        "table2": ["table2", "table2_0"],
    }


def test_read_table_from_hdf5_basic(mocker):
    """Test basic reading from HDF5 without units."""
    # Mock Table.read
    mock_table = mocker.MagicMock(spec=Table)
    mock_table.colnames = ["col1"]
    mocker.patch(ASTROPY_TABLE_READ, return_value=mock_table)

    # Mock h5py.File context
    mock_file = mocker.MagicMock()
    mock_dataset = mocker.MagicMock()
    mock_dataset.attrs = {}
    mock_file.__getitem__.return_value = mock_dataset
    mock_context = mocker.MagicMock()
    mock_context.__enter__.return_value = mock_file
    mocker.patch(H5PY_FILE, return_value=mock_context)

    result = read_table_from_hdf5(TEST_H5, TEST_TABLE_NAME)

    # Verify basic calls
    mock_file.__getitem__.assert_called_once_with(TEST_TABLE_NAME)
    assert result == mock_table


def test_read_table_from_hdf5_with_units(mocker):
    """Test reading from HDF5 with unit attributes."""
    # Mock Table.read
    mock_table = mocker.MagicMock(spec=Table)
    mock_table.colnames = ["col1"]
    mocker.patch(ASTROPY_TABLE_READ, return_value=mock_table)

    # Mock h5py.File context
    mock_file = mocker.MagicMock()
    mock_dataset = mocker.MagicMock()
    mock_dataset.attrs = {"col1_unit": "m"}
    mock_file.__getitem__.return_value = mock_dataset
    mock_context = mocker.MagicMock()
    mock_context.__enter__.return_value = mock_file
    mocker.patch(H5PY_FILE, return_value=mock_context)

    result = read_table_from_hdf5(TEST_H5, TEST_TABLE_NAME)

    # Verify unit assignment
    mock_table.__getitem__.assert_called_once_with("col1")
    assert result == mock_table


def test_read_table_from_hdf5_with_selected_columns(mocker):
    """Test reading only selected columns from a compound HDF5 dataset."""
    mock_file = mocker.MagicMock()
    mock_dataset = mocker.MagicMock()
    mock_dataset.dtype.names = ("col1", "col2")
    mock_dataset.attrs = {"col1_unit": "m"}

    selected_data = np.array([(1.0,), (2.0,)], dtype=[("col1", "f8")])
    fields_accessor = mocker.MagicMock()
    fields_accessor.__getitem__.return_value = selected_data
    mock_dataset.fields.return_value = fields_accessor

    mock_file.__getitem__.return_value = mock_dataset
    mock_context = mocker.MagicMock()
    mock_context.__enter__.return_value = mock_file
    mocker.patch(H5PY_FILE, return_value=mock_context)
    mock_table_read = mocker.patch(ASTROPY_TABLE_READ)

    result = read_table_from_hdf5(TEST_H5, TEST_TABLE_NAME, columns=["col1"])

    mock_table_read.assert_not_called()
    mock_dataset.fields.assert_called_once_with(["col1"])
    assert result.colnames == ["col1"]
    assert result["col1"].unit == u.Unit("m")


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ([("col1", "S3")], ["abc", "def"]),
        ([("col1", object)], ["abc", "def"]),
    ],
)
def test_read_table_from_hdf5_decodes_selected_string_columns(mocker, dtype, expected):
    """Test selected-column reads decode bytes-backed string columns."""
    mock_file = mocker.MagicMock()
    mock_dataset = mocker.MagicMock()
    mock_dataset.dtype.names = ("col1",)
    mock_dataset.attrs = {}

    selected_data = np.array([(b"abc",), (b"def",)], dtype=dtype)
    mock_dataset.fields.return_value = selected_data

    mock_file.__getitem__.return_value = mock_dataset
    mock_context = mocker.MagicMock()
    mock_context.__enter__.return_value = mock_file
    mocker.patch(H5PY_FILE, return_value=mock_context)

    result = read_table_from_hdf5(TEST_H5, TEST_TABLE_NAME, columns=["col1"])

    assert list(result["col1"]) == expected
    assert result["col1"].dtype.kind == "U"


def test_read_table_from_hdf5_with_missing_selected_columns(mocker):
    """Test selected-column read with unknown column names."""
    mock_file = mocker.MagicMock()
    mock_dataset = mocker.MagicMock()
    mock_dataset.dtype.names = ("col1",)
    mock_dataset.attrs = {}
    mock_file.__getitem__.return_value = mock_dataset
    mock_context = mocker.MagicMock()
    mock_context.__enter__.return_value = mock_file
    mocker.patch(H5PY_FILE, return_value=mock_context)

    with pytest.raises(KeyError, match="not found in table"):
        read_table_from_hdf5(TEST_H5, TEST_TABLE_NAME, columns=["col2"])
