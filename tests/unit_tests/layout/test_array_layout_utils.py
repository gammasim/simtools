#!/usr/bin/python3

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import astropy.units as u
import pytest
from astropy.table import QTable

from simtools.io import ascii_handler
from simtools.layout import array_layout_utils

# Constants for patch paths
PATCH_ASCII_COLLECT_FILE = "simtools.layout.array_layout_utils.ascii_handler.collect_data_from_file"
PATCH_SITEMODEL = "simtools.layout.array_layout_utils.SiteModel"


@pytest.fixture
def mock_read_table_from_file():
    return "simtools.layout.array_layout_utils.data_reader.read_table_from_file"


@pytest.fixture
def test_output():
    return "test_output.json"


@pytest.fixture
def mock_io_handler(test_output):
    with patch("simtools.layout.array_layout_utils.io_handler.IOHandler") as mock:
        instance = Mock()
        instance.get_output_file.return_value = test_output
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_model_data_writer():
    with patch("simtools.layout.array_layout_utils.ModelDataWriter") as mock:
        yield mock


@pytest.fixture
def mock_metadata_collector():
    with patch("simtools.layout.array_layout_utils.MetadataCollector") as mock:
        yield mock


@pytest.fixture
def mock_array_model():
    with patch("simtools.layout.array_layout_utils.ArrayModel") as mock:
        yield mock


@pytest.fixture
def test_path():
    return "/test/path"


def test_write_array_layouts(
    mock_io_handler, mock_model_data_writer, mock_metadata_collector, test_path, test_output
):
    array_layouts = {"value": [{"name": "test_array", "elements": ["tel1", "tel2"]}]}
    args_dict = {
        "site": "North",
        "output_path": test_path,
        "updated_parameter_version": "v1",
    }
    array_layout_utils.write_array_layouts(array_layouts, args_dict)

    mock_io_handler.return_value.set_paths.assert_called_once_with(output_path=test_path)
    mock_io_handler.return_value.get_output_file.assert_called_once_with("array-layouts-v1.json")

    mock_model_data_writer.dump_model_parameter.assert_called_once_with(
        parameter_name="array_layouts",
        value=array_layouts["value"],
        instrument="OBS-North",
        parameter_version="v1",
        output_file=test_output,
    )

    mock_metadata_collector.dump.assert_called_once_with(
        args_dict, test_output, add_activity_name=True
    )


def test_merge_array_layouts():
    """Test merging of array layouts."""
    # Define test inputs
    layouts_1 = {
        "value": [
            {"name": "array1", "elements": ["tel1", "tel2"]},
            {"name": "array2", "elements": ["tel3", "tel4"]},
        ]
    }

    layouts_2 = [
        {"name": "array3", "elements": ["tel1", "tel2"]},  # Same elements as array1
        {"name": "array4", "elements": ["tel5", "tel6"]},  # New elements
    ]

    # Call function
    merged = array_layout_utils.merge_array_layouts(layouts_1, layouts_2)

    # Assert results
    assert len(merged["value"]) == 3

    # Check renamed layout (array1 -> array3)
    assert any(
        layout["name"] == "array3" and sorted(layout["elements"]) == ["tel1", "tel2"]
        for layout in merged["value"]
    )

    # Check unchanged layout
    assert any(
        layout["name"] == "array2" and sorted(layout["elements"]) == ["tel3", "tel4"]
        for layout in merged["value"]
    )

    # Check newly added layout
    assert any(
        layout["name"] == "array4" and sorted(layout["elements"]) == ["tel5", "tel6"]
        for layout in merged["value"]
    )


def test_get_ctao_array_element_name():
    """Test getting array element name from common identifier."""
    # Test normal case
    array_element_ids = {
        "array_elements": [{"id": "T1", "name": "telescope1"}, {"id": "T2", "name": "telescope2"}]
    }

    assert array_layout_utils._get_ctao_array_element_name("T1", array_element_ids) == "telescope1"
    assert array_layout_utils._get_ctao_array_element_name("T2", array_element_ids) == "telescope2"

    # Test non-existent id
    assert array_layout_utils._get_ctao_array_element_name("T3", array_element_ids) is None

    # Test empty array elements
    empty_elements = {"array_elements": []}
    assert array_layout_utils._get_ctao_array_element_name("T1", empty_elements) is None

    # Test missing array_elements key
    empty_dict = {}
    assert array_layout_utils._get_ctao_array_element_name("T1", empty_dict) is None


@patch("simtools.layout.array_layout_utils.names")
def test_get_ctao_layouts_per_site(mock_names):
    """Test getting array layouts per site."""
    # Mock site determination function
    mock_names.get_site_from_array_element_name.side_effect = lambda x: (
        "north" if "N" in x else "south"
    )

    # Test data
    site = "north"
    sub_arrays = {
        "subarrays": [
            {"name": "array1", "array_element_ids": ["T1", "T2"]},
            {"name": "array2", "array_element_ids": ["T3", "T4"]},
        ]
    }
    array_element_ids = {
        "array_elements": [
            {"id": "T1", "name": "N_tel1"},
            {"id": "T2", "name": "N_tel2"},
            {"id": "T3", "name": "S_tel1"},
            {"id": "T4", "name": "S_tel2"},
        ]
    }

    # Call function
    layouts = array_layout_utils._get_ctao_layouts_per_site(site, sub_arrays, array_element_ids)

    # Assert results
    assert len(layouts) == 1
    assert layouts[0]["name"] == "array1"
    assert layouts[0]["elements"] == ["N_tel1", "N_tel2"]

    # Test empty subarrays
    layouts = array_layout_utils._get_ctao_layouts_per_site(
        site, {"subarrays": []}, array_element_ids
    )
    assert len(layouts) == 0

    # Test missing keys
    layouts = array_layout_utils._get_ctao_layouts_per_site(site, {}, array_element_ids)
    assert len(layouts) == 0

    # Test array with no matching elements
    site = "south"
    layouts = array_layout_utils._get_ctao_layouts_per_site(
        site,
        {"subarrays": [{"name": "array1", "array_element_ids": ["T1", "T2"]}]},
        array_element_ids,
    )
    assert len(layouts) == 0


def test_retrieve_ctao_array_layouts_from_url():
    """Test retrieving array layouts from URL."""
    with (
        patch("simtools.layout.array_layout_utils.gen") as mock_gen,
        patch(
            "simtools.layout.array_layout_utils.ascii_handler.collect_data_from_http"
        ) as mock_ascii_handler,
    ):
        mock_gen.is_url.return_value = True
        mock_ascii_handler.return_value = {"subarrays": [], "array_elements": []}

        array_layout_utils.retrieve_ctao_array_layouts(
            site="north", repository_url="https://test.com", branch_name="test-branch"
        )

        mock_gen.is_url.assert_called_once_with("https://test.com")
        mock_ascii_handler.assert_called_with(url="https://test.com/test-branch/subarray-ids.json")


def test_retrieve_ctao_array_layouts_from_file(test_path):
    """Test retrieving array layouts from local file."""
    with (
        patch("simtools.layout.array_layout_utils.gen") as mock_gen,
        patch(PATCH_ASCII_COLLECT_FILE) as mock_ascii_handler,
    ):
        mock_gen.is_url.return_value = False
        mock_ascii_handler.return_value = {"subarrays": [], "array_elements": []}

        array_layout_utils.retrieve_ctao_array_layouts(
            site="north", repository_url=test_path, branch_name="test-branch"
        )

        mock_gen.is_url.assert_called_once_with(test_path)
        assert mock_ascii_handler.call_count == 2


def test_validate_array_layouts_with_db_valid():
    """Test validation with valid array layouts."""
    production_table = {"parameters": {"tel1": {}, "tel2": {}, "tel3": {}, "tel4": {}}}

    array_layouts = {
        "value": [
            {"name": "array1", "elements": ["tel1", "tel2"]},
            {"name": "array2", "elements": ["tel3", "tel4"]},
        ]
    }

    result = array_layout_utils.validate_array_layouts_with_db(production_table, array_layouts)
    assert result == array_layouts


def test_validate_array_layouts_with_db_invalid():
    """Test validation with invalid array layouts."""
    production_table = {"parameters": {"tel1": {}, "tel2": {}}}

    array_layouts = {
        "value": [
            {"name": "array1", "elements": ["tel1", "tel2"]},
            {"name": "array2", "elements": ["tel3", "tel4"]},  # tel3, tel4 not in DB
        ]
    }

    with pytest.raises(ValueError, match=r"Invalid array elements found: \['tel3', 'tel4'\]"):
        array_layout_utils.validate_array_layouts_with_db(production_table, array_layouts)


def test_validate_array_layouts_with_db_empty_production_table():
    """Test validation with empty production table."""
    production_table = {"parameters": {}}

    array_layouts = {
        "value": [
            {"name": "array1", "elements": ["tel1"]},
        ]
    }

    with pytest.raises(ValueError, match=r"Invalid array elements found: \['tel1'\]"):
        array_layout_utils.validate_array_layouts_with_db(production_table, array_layouts)


def test_validate_array_layouts_with_db_empty_array_layouts():
    """Test validation with empty array layouts."""
    production_table = {"parameters": {"tel1": {}, "tel2": {}}}

    array_layouts = {"value": []}

    result = array_layout_utils.validate_array_layouts_with_db(production_table, array_layouts)
    assert result == array_layouts


def test_validate_array_layouts_with_db_missing_keys():
    """Test validation with missing keys in dictionaries."""
    # Missing parameters key
    production_table = {}
    array_layouts = {
        "value": [
            {"name": "array1", "elements": ["tel1"]},
        ]
    }

    with pytest.raises(ValueError, match=r"Invalid array elements found: \['tel1'\]"):
        array_layout_utils.validate_array_layouts_with_db(production_table, array_layouts)

    # Missing value key
    production_table = {"parameters": {"tel1": {}}}
    array_layouts = {}

    result = array_layout_utils.validate_array_layouts_with_db(production_table, array_layouts)
    assert result == array_layouts

    # Missing elements key in layout
    array_layouts = {
        "value": [
            {"name": "array1"},  # No elements key
        ]
    }

    result = array_layout_utils.validate_array_layouts_with_db(production_table, array_layouts)
    assert result == array_layouts


def test_validate_array_layouts_with_db_partial_invalid():
    """Test validation with partially invalid array layouts."""
    production_table = {"parameters": {"tel1": {}, "tel2": {}}}

    array_layouts = {
        "value": [
            {"name": "array1", "elements": ["tel1", "tel2"]},  # Valid
            {"name": "array2", "elements": ["tel1", "tel3"]},  # tel3 invalid
        ]
    }

    with pytest.raises(ValueError, match=r"Invalid array elements found: \['tel3'\]"):
        array_layout_utils.validate_array_layouts_with_db(production_table, array_layouts)


def test_get_array_layouts_from_parameter_file_valid(mocker, mock_array_model):
    model_version = "6.0.0"
    fake_data = {
        "value": [
            {"name": "array1"},
            {"name": "array2"},
        ],
        "site": "north",
    }
    mocker.patch(
        PATCH_ASCII_COLLECT_FILE,
        return_value=fake_data,
    )
    fake_table = ["tel1", "tel2"]
    instance = mock_array_model.return_value
    instance.export_array_elements_as_table.return_value = fake_table

    results = array_layout_utils.get_array_layouts_from_parameter_file(
        "test_file.json", model_version
    )

    assert isinstance(results, list)
    assert len(results) == 2
    for layout in results:
        assert layout["site"] == "north"
        assert layout["array_elements"] == fake_table
        assert layout["name"] in ["array1", "array2"]

    expected_calls = [
        mocker.call(
            model_version=model_version,
            site="north",
            array_elements=None,
            layout_name="array1",
        ),
        mocker.call(
            model_version=model_version,
            site="north",
            array_elements=None,
            layout_name="array2",
        ),
    ]
    mock_array_model.assert_has_calls(expected_calls, any_order=True)


def test_get_array_layouts_from_parameter_file_missing_value_key(mocker):
    fake_data = {
        "site": "north",
    }
    # Patch ascii_handler.collect_data_from_file to return fake_data without the "value" key.
    mocker.patch(
        PATCH_ASCII_COLLECT_FILE,
        return_value=fake_data,
    )

    with pytest.raises(ValueError, match=r"Missing 'value' key in layout file."):
        array_layout_utils.get_array_layouts_from_parameter_file("test_file.json", "6.0.0")


def test_get_array_layouts_from_db_with_layout_name(mock_array_model):
    # Test when a specific layout_name is provided.
    layout_name = "layout_test"
    site = "North"
    model_version = "6.0.0"
    fake_table = ["tel1", "tel2"]

    # Patch ArrayModel so that export_array_elements_as_table returns fake_table.
    instance = MagicMock()
    instance.export_array_elements_as_table.return_value = fake_table
    mock_array_model.return_value = instance

    # Call the function with layout_name provided.
    result = array_layout_utils.get_array_layouts_from_db(layout_name, site, model_version)

    # Expected: a list with one dict corresponding to the provided layout_name.
    expected = {
        "name": layout_name,
        "site": site,
        "array_elements": fake_table,
    }

    # Assert that ArrayModel was initialized with the correct parameters.
    mock_array_model.assert_called_once_with(
        model_version=model_version,
        site=site,
        array_elements=None,
        layout_name=layout_name,
    )
    instance.export_array_elements_as_table.assert_called_once_with(coordinate_system="ground")
    assert result == expected


def test_get_array_layouts_from_db_without_layout_name(mocker, mock_array_model):
    # Test when layout_name is None, so SiteModel is used to retrieve layout names.
    layout_name = None
    site = "South"
    model_version = "7.0.0"
    # Fake layout names returned by SiteModel.
    fake_layout_names = ["layout1", "layout2"]

    # Patch SiteModel so that get_list_of_array_layouts returns our fake_layout_names.
    mock_site_model = mocker.patch(PATCH_SITEMODEL)
    instance_site = MagicMock()
    instance_site.get_list_of_array_layouts.return_value = fake_layout_names
    mock_site_model.return_value = instance_site

    # Create separate fake return values for each layout.
    fake_table1 = ["telA", "telB"]
    fake_table2 = ["telC", "telD"]

    # The side_effect will help return different instances based on the call.
    def array_model_side_effect(*args, **kwargs):
        instance = MagicMock()
        if kwargs.get("layout_name") == "layout1":
            instance.export_array_elements_as_table.return_value = fake_table1
        elif kwargs.get("layout_name") == "layout2":
            instance.export_array_elements_as_table.return_value = fake_table2
        return instance

    mock_array_model.side_effect = array_model_side_effect

    # Call the function with layout_name as None.
    result = array_layout_utils.get_array_layouts_from_db(layout_name, site, model_version)

    # Expected: a list with two dicts.
    expected = [
        {"name": "layout1", "site": site, "array_elements": fake_table1},
        {"name": "layout2", "site": site, "array_elements": fake_table2},
    ]

    # Assert that SiteModel was correctly used.
    mock_site_model.assert_called_once_with(site=site, model_version=model_version)
    instance_site.get_list_of_array_layouts.assert_called_once()

    # Assert that ArrayModel was called for each layout returned by SiteModel.
    calls = [
        mocker.call(
            model_version=model_version,
            site=site,
            array_elements=None,
            layout_name="layout1",
        ),
        mocker.call(
            model_version=model_version,
            site=site,
            array_elements=None,
            layout_name="layout2",
        ),
    ]
    mock_array_model.assert_has_calls(calls, any_order=True)

    # Assert the result matches the expected output.
    assert result == expected


def test_get_array_layouts_using_telescope_lists_from_db_with_site(mocker, mock_array_model):
    telescope_lists = [["tel1", "tel2"], ["tel3", "tel4"]]
    site = "North"
    fake_table = ["fake", "elements"]

    # Patch ArrayModel to return a fake table via export_array_elements_as_table.
    instance = MagicMock()
    instance.export_array_elements_as_table.return_value = fake_table
    mock_array_model.return_value = instance

    results = array_layout_utils.get_array_layouts_using_telescope_lists_from_db(
        telescope_lists, site, "6.0.0", coordinate_system="ground"
    )

    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert result["name"] == "list"
        assert result["site"] == site
        assert result["array_elements"] == fake_table

    assert mock_array_model.call_count == 2


def test_get_array_layouts_using_telescope_lists_from_db_without_site_single(
    mocker, mock_array_model
):
    # Case where site is None and all telescope list elements originate from the same site.
    telescope_lists = [["N_tel1", "N_tel2"]]
    site = None
    fake_table = ["fake", "elements"]

    # Patch names.get_site_from_array_element_name to always return 'north'.
    mock_names = mocker.patch("simtools.layout.array_layout_utils.names")
    mock_names.get_site_from_array_element_name.return_value = "north"

    instance = MagicMock()
    instance.export_array_elements_as_table.return_value = fake_table
    mock_array_model.return_value = instance

    results = array_layout_utils.get_array_layouts_using_telescope_lists_from_db(
        telescope_lists, site, "6.1.0", coordinate_system="ground"
    )

    assert isinstance(results, list)
    assert len(results) == 1
    result = results[0]
    assert result["name"] == "list"
    assert result["site"] == "north"
    assert result["array_elements"] == fake_table


def test_get_array_layouts_using_telescope_lists_from_db_without_site_multiple_error(mocker):
    # Case where site is None and telescope list elements come from different sites.
    telescope_lists = [["N_tel1", "S_tel1"]]

    def fake_get_site(name):
        return "North" if "N" in name else "South"

    mock_names = mocker.patch("simtools.layout.array_layout_utils.names")
    mock_names.get_site_from_array_element_name.side_effect = fake_get_site

    with pytest.raises(ValueError, match="Telescope list contains elements from multiple sites:"):
        array_layout_utils.get_array_layouts_using_telescope_lists_from_db(
            telescope_lists, None, "6.2.0", coordinate_system="ground"
        )


def test_get_array_layouts_from_file_single_string(mocker, mock_read_table_from_file):
    fake_table = ["dummy_table"]
    mocker.patch(mock_read_table_from_file, return_value=fake_table)
    file_path = "dummy_file.txt"
    layouts = array_layout_utils.get_array_layouts_from_file(file_path)
    assert len(layouts) == 1
    expected_name = "dummy_file"  # from "dummy_file.txt"
    assert layouts[0]["name"] == expected_name
    assert layouts[0]["array_elements"] == fake_table


def test_get_array_layouts_from_file_single_path(mocker, mock_read_table_from_file):
    fake_table = ["path_table"]
    mocker.patch(mock_read_table_from_file, return_value=fake_table)
    file_path = Path("example_file.dat")
    layouts = array_layout_utils.get_array_layouts_from_file(file_path)
    assert len(layouts) == 1
    expected_name = "example_file"  # from "example_file.dat"
    assert layouts[0]["name"] == expected_name
    assert layouts[0]["array_elements"] == fake_table


def test_get_array_layouts_from_file_list(mocker, mock_read_table_from_file):
    fake_table1 = ["table1"]
    fake_table2 = ["table2"]
    mocker.patch(mock_read_table_from_file, side_effect=[fake_table1, fake_table2])
    file_paths = ["file1.csv", "file2.csv"]
    layouts = array_layout_utils.get_array_layouts_from_file(file_paths)
    assert len(layouts) == 2
    assert layouts[0]["name"] == "file1"  # from "file1.csv"
    assert layouts[1]["name"] == "file2"  # from "file2.csv"
    assert layouts[0]["array_elements"] == fake_table1
    assert layouts[1]["array_elements"] == fake_table2


def test_get_array_layout_dict_with_layout_name(mock_array_model):
    """Test _get_array_layout_dict with a layout name provided."""
    # Setup test data
    model_version = "6.0.0"
    site = "north"
    layout_name = "test_layout"
    fake_table = ["tel1", "tel2"]

    # Mock ArrayModel instance
    instance = MagicMock()
    instance.export_array_elements_as_table.return_value = fake_table
    mock_array_model.return_value = instance

    # Call function
    result = array_layout_utils._get_array_layout_dict(
        model_version, site, None, layout_name, "ground"
    )

    # Verify ArrayModel initialization
    mock_array_model.assert_called_once_with(
        model_version=model_version,
        site=site,
        array_elements=None,
        layout_name=layout_name,
    )

    # Verify export_array_elements_as_table call
    instance.export_array_elements_as_table.assert_called_once_with(coordinate_system="ground")

    # Check result
    assert result == {"name": layout_name, "site": site, "array_elements": fake_table}


def test_get_array_layout_dict_with_telescope_list(mock_array_model):
    """Test _get_array_layout_dict with a telescope list provided."""
    # Setup test data
    model_version = "6.0.0"
    site = "south"
    telescope_list = ["tel1", "tel2", "tel3"]
    fake_table = ["tel_data1", "tel_data2", "tel_data3"]

    # Mock ArrayModel instance
    instance = MagicMock()
    instance.export_array_elements_as_table.return_value = fake_table
    mock_array_model.return_value = instance

    # Call function
    result = array_layout_utils._get_array_layout_dict(
        model_version, site, telescope_list, None, "ground"
    )

    # Verify ArrayModel initialization
    mock_array_model.assert_called_once_with(
        model_version=model_version,
        site=site,
        array_elements=telescope_list,
        layout_name=None,
    )

    # Verify export_array_elements_as_table call
    instance.export_array_elements_as_table.assert_called_once_with(coordinate_system="ground")

    # Check result
    assert result == {"name": "list", "site": site, "array_elements": fake_table}


def test_read_array_layouts_from_db_specific_layouts(mocker):
    """Test _read_array_layouts_from_db with specific layout names."""
    mock_site_model = mocker.patch(PATCH_SITEMODEL)
    instance = mock_site_model.return_value
    instance.get_array_elements_for_layout.side_effect = lambda name: (
        [1, 2] if name == "LST" else [3, 4]
    )

    layouts = ["LST", "MST"]
    site = "North"
    model_version = "v1.0.0"

    result = array_layout_utils.get_array_elements_from_db_for_layouts(layouts, site, model_version)

    assert result == {"LST": [1, 2], "MST": [3, 4]}
    mock_site_model.assert_called_once_with(site=site, model_version=model_version)
    assert instance.get_array_elements_for_layout.call_count == 2
    instance.get_array_elements_for_layout.assert_any_call("LST")
    instance.get_array_elements_for_layout.assert_any_call("MST")


def test_read_array_layouts_from_db_all_layouts(mocker):
    """Test _read_array_layouts_from_db with 'all' layouts."""
    mock_site_model = mocker.patch("simtools.layout.array_layout_utils.SiteModel")
    instance = mock_site_model.return_value
    instance.get_list_of_array_layouts.return_value = ["LST", "MST"]
    instance.get_array_elements_for_layout.side_effect = lambda name: (
        [10, 20] if name == "LST" else [30, 40]
    )

    layouts = ["all"]
    site = "South"
    model_version = "v2.0.0"

    result = array_layout_utils.get_array_elements_from_db_for_layouts(layouts, site, model_version)

    assert result == {"LST": [10, 20], "MST": [30, 40]}
    instance.get_list_of_array_layouts.assert_called_once()
    assert instance.get_array_elements_for_layout.call_count == 2
    instance.get_array_elements_for_layout.assert_any_call("LST")
    instance.get_array_elements_for_layout.assert_any_call("MST")


@pytest.fixture
def minimal_args_dict():
    return {
        "array_layout_name_background": None,
        "array_layout_name": None,
        "plot_all_layouts": False,
        "array_layout_parameter_file": None,
        "array_layout_file": None,
        "array_element_list": None,
        "site": "North",
        "model_version": "1.0.0",
        "coordinate_system": "ground",
    }


def test_read_layouts_returns_empty_lists_when_no_inputs(minimal_args_dict):
    layouts, background = array_layout_utils.read_layouts(minimal_args_dict)
    assert layouts == []
    assert background is None


def test_read_layouts_with_array_layout_name_background(minimal_args_dict):
    args = minimal_args_dict.copy()
    args["array_layout_name_background"] = "bg_layout"
    args["array_layout_name"] = "main_layout"
    with patch("simtools.layout.array_layout_utils.get_array_layouts_from_db") as mock_get:
        mock_get.side_effect = [
            {"array_elements": ["tel1", "tel2"]},
            {"name": "main_layout", "site": "North", "array_elements": ["tel3", "tel4"]},
        ]
        layouts, background = array_layout_utils.read_layouts(args)
        assert background == ["tel1", "tel2"]
        assert isinstance(layouts, list)
        assert layouts[0]["name"] == "main_layout"
        # Assert get_array_layouts_from_db was called twice with expected arguments
        expected_calls = [
            (
                (
                    args["array_layout_name_background"],
                    args["site"],
                    args["model_version"],
                    args["coordinate_system"],
                ),
            ),
            (
                (
                    args["array_layout_name"],
                    args["site"],
                    args["model_version"],
                    args["coordinate_system"],
                ),
            ),
        ]
        actual_calls = mock_get.call_args_list
        assert len(actual_calls) == 2
        for actual, expected in zip(actual_calls, expected_calls):
            assert actual[0] == expected[0]


def test_read_layouts_with_plot_all_layouts(minimal_args_dict):
    args = minimal_args_dict.copy()
    args["plot_all_layouts"] = True
    with patch("simtools.layout.array_layout_utils.get_array_layouts_from_db") as mock_get:
        mock_get.return_value = [{"name": "layout1", "array_elements": ["tel1"]}]
        layouts, background = array_layout_utils.read_layouts(args)
        assert isinstance(layouts, list)
        assert layouts[0]["name"] == "layout1"
        assert background is None


def test_read_layouts_with_array_layout_parameter_file(minimal_args_dict):
    args = minimal_args_dict.copy()
    args["array_layout_parameter_file"] = "param_file.json"
    with patch(
        "simtools.layout.array_layout_utils.get_array_layouts_from_parameter_file"
    ) as mock_get:
        mock_get.return_value = [{"name": "layout_param", "array_elements": ["telA"]}]
        layouts, background = array_layout_utils.read_layouts(args)
        assert isinstance(layouts, list)
        assert layouts[0]["name"] == "layout_param"
        assert background is None


def test_read_layouts_with_array_layout_file(minimal_args_dict):
    args = minimal_args_dict.copy()
    args["array_layout_file"] = "layout_file.txt"
    with patch("simtools.layout.array_layout_utils.get_array_layouts_from_file") as mock_get:
        mock_get.return_value = [{"name": "layout_file", "array_elements": ["telB"]}]
        layouts, background = array_layout_utils.read_layouts(args)
        assert isinstance(layouts, list)
        assert layouts[0]["name"] == "layout_file"
        assert background is None


def test_read_layouts_with_array_element_list(minimal_args_dict):
    args = minimal_args_dict.copy()
    args["array_element_list"] = ["telC", "telD"]
    with patch(
        "simtools.layout.array_layout_utils.get_array_layouts_using_telescope_lists_from_db"
    ) as mock_get:
        mock_get.return_value = [{"name": "list", "array_elements": ["telC", "telD"]}]
        layouts, background = array_layout_utils.read_layouts(args)
        assert isinstance(layouts, list)
        assert layouts[0]["name"] == "list"
        assert background is None


def test_create_regular_array_simple():
    """Test creating a regular array with a single telescope."""
    table = array_layout_utils.create_regular_array(
        "1MST", "North", n_telescopes=1, telescope_type="MST", telescope_distance=100 * u.m
    )
    assert len(table) == 1
    assert table.meta["array_name"] == "1MST"
    assert table.meta["site"] == "North"
    assert table["position_x"][0].value == 0
    assert table["position_y"][0].value == 0
    assert table["position_z"][0].value == 0


def test_create_regular_array_four_telescopes_square(mocker):
    """Test creating a square regular array with four telescopes."""
    mocker.patch(
        "simtools.layout.array_layout_utils.names.generate_array_element_name_from_type_site_id",
        side_effect=lambda tel_type, site, idx: f"{tel_type}_{site}_{idx}",
    )
    table = array_layout_utils.create_regular_array(
        "4MST", "South", n_telescopes=4, telescope_type="MST", telescope_distance=120 * u.m
    )
    assert len(table) == 4
    assert table.meta["array_name"] == "4MST"
    assert table.meta["site"] == "South"
    assert set(table["telescope_name"]) == {f"MST_South_0{i}" for i in range(1, 5)}
    for x, y, z in zip(table["position_x"], table["position_y"], table["position_z"]):
        assert abs(abs(x.value) - 120) < 1e-6 or abs(x.value) < 1e-6
        assert abs(abs(y.value) - 120) < 1e-6 or abs(y.value) < 1e-6
        assert z.value == 0


def test_create_regular_array_star_shape(mocker):
    """Test creating a star-shaped regular array with four telescopes."""
    mocker.patch(
        "simtools.layout.array_layout_utils.names.generate_array_element_name_from_type_site_id",
        side_effect=lambda tel_type, site, idx: f"{tel_type}_{site}_{idx}",
    )
    table = array_layout_utils.create_regular_array(
        "4LST",
        "North",
        n_telescopes=4,
        telescope_type="LST",
        telescope_distance=150 * u.m,
        shape="star",
    )
    assert len(table) == 4
    assert table.meta["array_name"] == "4LST"
    assert set(table["telescope_name"]) == {f"LST_North_0{i}" for i in range(1, 5)}
    assert all(z.value == 0 for z in table["position_z"])


def test_create_regular_array_errors():
    """Test that create_regular_array raises appropriate errors."""
    distance = 100 * u.m

    with pytest.raises(ValueError, match="Unsupported number of telescopes"):
        array_layout_utils.create_regular_array(
            "5MST", "South", n_telescopes=5, telescope_type="MST", telescope_distance=distance
        )

    with pytest.raises(ValueError, match="Unsupported array shape: circle"):
        array_layout_utils.create_regular_array(
            "1MST",
            "North",
            n_telescopes=1,
            telescope_type="MST",
            telescope_distance=distance,
            shape="circle",
        )


def test_create_regular_array_metadata():
    """Test that create_regular_array preserves units and sorts telescope names."""
    table = array_layout_utils.create_regular_array(
        "4MST", "South", n_telescopes=4, telescope_type="MST", telescope_distance=120 * u.m
    )
    assert table["position_x"].unit == u.m
    assert table["position_y"].unit == u.m
    assert table["position_z"].unit == u.m
    assert list(table["telescope_name"]) == sorted(table["telescope_name"])


def test_get_array_name_valid():
    assert array_layout_utils._get_array_name("4MST") == ("MST", 4)
    assert array_layout_utils._get_array_name("1LST") == ("LST", 1)
    assert array_layout_utils._get_array_name("2SST") == ("SST", 2)


def test_get_array_name_invalid():
    with pytest.raises(ValueError, match="Invalid array_name: 'MST'"):
        array_layout_utils._get_array_name("MST")
    with pytest.raises(ValueError, match="Invalid array_name: 'A4MST'"):
        array_layout_utils._get_array_name("A4MST")


def test_write_array_elements_from_file_to_repository_utm(tmp_test_directory):
    array_layout_utils.write_array_elements_from_file_to_repository(
        coordinate_system="utm",
        input_file="tests/resources/telescope_positions-North-utm.ecsv",
        repository_path=tmp_test_directory,
        parameter_version="5.7.0",
    )
    output = Path(tmp_test_directory, "MSTN-03/array_element_position_utm.json")

    assert output.exists()

    para = ascii_handler.collect_data_from_file(output)
    assert para["parameter"] == "array_element_position_utm"
    assert para["parameter_version"] == "5.7.0"
    assert para["instrument"] == "MSTN-03"
    assert para["unit"] == "m"
    assert para["value"][0] == pytest.approx(217401.1)


def test_write_array_elements_from_file_to_repository_ground(tmp_test_directory):
    array_layout_utils.write_array_elements_from_file_to_repository(
        coordinate_system="ground",
        input_file="tests/resources/telescope_positions-North-ground.ecsv",
        repository_path=tmp_test_directory,
        parameter_version="5.7.0",
    )
    output = Path(tmp_test_directory, "MSTN-03/array_element_position_ground.json")
    assert output.exists()

    para = ascii_handler.collect_data_from_file(output)
    assert para["parameter"] == "array_element_position_ground"
    assert para["parameter_version"] == "5.7.0"
    assert para["instrument"] == "MSTN-03"
    assert para["unit"] == "m"
    assert para["value"][0] == pytest.approx(26.86)


def test_write_array_elements_from_file_to_repository_error(tmp_test_directory):
    with pytest.raises(
        ValueError, match=r"Unsupported coordinate system: invalid. Allowed are 'utm' and 'ground'."
    ):
        array_layout_utils.write_array_elements_from_file_to_repository(
            coordinate_system="invalid",
            input_file="tests/resources/telescope_positions-North-ground.ecsv",
            repository_path=tmp_test_directory,
            parameter_version="5.7.0",
        )


@pytest.fixture
def mock_telescope_names(mocker):
    """Fixture for mocking telescope name generation."""
    return mocker.patch(
        "simtools.layout.array_layout_utils.names.generate_array_element_name_from_type_site_id",
        side_effect=lambda tel_type, site, idx: f"{tel_type}_{site}_{idx}",
    )


def test_create_star_array_positions(mock_telescope_names):
    """Test _create_star_array creates correct positions for multiple telescope counts."""
    test_cases = [
        (1, [(100, 0)]),
        (4, [(120, 0), (0, 120), (-120, 0), (0, -120)]),
        (8, [(150, 0), (0, 150), (-150, 0), (0, -150), (300, 0), (0, 300), (-300, 0), (0, -300)]),
    ]

    for n_tel, expected_positions in test_cases:
        tel_name, pos_x, pos_y, pos_z = [], [], [], []
        distance = expected_positions[0][0] * u.m if expected_positions[0][0] != 0 else 100 * u.m

        array_layout_utils._create_star_array(
            tel_name, pos_x, pos_y, pos_z, n_tel, "MST", "North", distance
        )

        assert len(tel_name) == n_tel
        for i, (exp_x, exp_y) in enumerate(expected_positions):
            assert pos_x[i] == exp_x * u.m
            assert pos_y[i] == exp_y * u.m
            assert pos_z[i] == 0 * u.m


def test_create_square_array_positions(mock_telescope_names):
    """Test _create_square_array creates correct positions."""
    test_cases = [
        (1, [(0, 0)]),
        (4, [(120, -120), (-120, 120), (-120, -120), (120, 120)]),
    ]

    for n_tel, expected_positions in test_cases:
        tel_name, pos_x, pos_y, pos_z = [], [], [], []
        distance = 120 * u.m

        array_layout_utils._create_square_array(
            tel_name, pos_x, pos_y, pos_z, n_tel, "MST", "South", distance
        )

        assert len(tel_name) == n_tel
        for i, (exp_x, exp_y) in enumerate(expected_positions):
            assert pos_x[i] == exp_x * u.m
            assert pos_y[i] == exp_y * u.m
            assert pos_z[i] == 0 * u.m


def test_create_square_array_errors(mocker):
    """Test _create_square_array with unsupported number of telescopes."""
    mocker.patch(
        "simtools.layout.array_layout_utils.names.generate_array_element_name_from_type_site_id"
    )
    tel_name, pos_x, pos_y, pos_z = [], [], [], []
    distance = 100 * u.m

    for n_tel in [2, 5]:
        with pytest.raises(
            ValueError, match=f"Unsupported number of telescopes for square array: {n_tel}"
        ):
            array_layout_utils._create_square_array(
                tel_name, pos_x, pos_y, pos_z, n_tel, "MST", "North", distance
            )


def test_write_array_elements_info_yaml_basic(tmp_path):
    """Test writing array elements info YAML file with basic functionality."""
    array_table = QTable(
        {
            "telescope_name": ["MST_North_01", "MST_North_02"],
            "position_x": [100, -100] * u.m,
            "position_y": [100, -100] * u.m,
            "position_z": [0, 0] * u.m,
        },
        meta={"array_name": "2MST", "site": "North"},
    )
    output_file = tmp_path / "test_layout.yaml"

    array_layout_utils.write_array_elements_info_yaml(
        array_table, "North", "6.0.0", output_file, "2.0.0"
    )

    assert output_file.exists()
    data = ascii_handler.collect_data_from_file(output_file)

    assert data["model_version"] == "6.0.0"
    assert data["model_update"] == "patch_update"
    assert "6.0.0" in data["model_version_history"]
    assert "2MST" in data["description"]

    array_layouts = data["changes"]["OBS-North"]["array_layouts"]
    assert array_layouts["version"] == "2.0.0"
    assert array_layouts["value"][0]["name"] == "2MST"
    assert set(array_layouts["value"][0]["elements"]) == {"MST_North_01", "MST_North_02"}

    tel_01 = data["changes"]["MST_North_01"]["array_element_position_ground"]
    assert tel_01["version"] == "2.0.0"
    assert tel_01["value"] == [100.0, 100.0, 0.0]
    assert tel_01["unit"] == "m"


def test_write_array_elements_info_yaml_custom_values(tmp_path):
    """Test YAML writing with various custom values."""
    test_cases = [
        {
            "name": "single_telescope",
            "telescopes": ["LST_South_01"],
            "positions": ([0], [0], [0]),
            "site": "South",
            "model_version": "7.0.0",
            "parameter_version": "3.0.0",
        },
        {
            "name": "fractional_positions",
            "telescopes": ["MST_North_01", "MST_North_02"],
            "positions": ([123.456, -234.567], [789.012, -456.789], [1.234, 2.345]),
            "site": "North",
            "model_version": "6.0.0",
            "parameter_version": "2.0.0",
        },
    ]

    for case in test_cases:
        array_table = QTable(
            {
                "telescope_name": case["telescopes"],
                "position_x": case["positions"][0] * u.m,
                "position_y": case["positions"][1] * u.m,
                "position_z": case["positions"][2] * u.m,
            },
            meta={"array_name": f"{len(case['telescopes'])}TEL", "site": case["site"]},
        )

        output_file = tmp_path / f"{case['name']}.yaml"
        array_layout_utils.write_array_elements_info_yaml(
            array_table, case["site"], case["model_version"], output_file, case["parameter_version"]
        )

        data = ascii_handler.collect_data_from_file(output_file)
        assert data["model_version"] == case["model_version"]
        obs_key = f"OBS-{case['site']}"
        assert data["changes"][obs_key]["array_layouts"]["version"] == case["parameter_version"]

        for tel_name in case["telescopes"]:
            tel_data = data["changes"][tel_name]["array_element_position_ground"]
            assert tel_data["version"] == case["parameter_version"]


def test_get_array_elements_from_db_for_layouts_specific_layouts(mocker):
    """Test get_array_elements_from_db_for_layouts with specific layout names."""
    mock_site_model = mocker.patch(PATCH_SITEMODEL)
    instance = mock_site_model.return_value
    instance.get_array_elements_for_layout.side_effect = lambda name: (
        ["tel1", "tel2"] if name == "LST" else ["tel3", "tel4"]
    )

    layouts = ["LST", "MST"]
    site = "North"
    model_version = "6.0.0"

    result = array_layout_utils.get_array_elements_from_db_for_layouts(layouts, site, model_version)

    assert result == {"LST": ["tel1", "tel2"], "MST": ["tel3", "tel4"]}
    mock_site_model.assert_called_once_with(site=site, model_version=model_version)
    assert instance.get_array_elements_for_layout.call_count == 2
    instance.get_array_elements_for_layout.assert_any_call("LST")
    instance.get_array_elements_for_layout.assert_any_call("MST")


def test_get_array_elements_from_db_for_layouts_all_layouts(mocker):
    """Test get_array_elements_from_db_for_layouts with 'all' layouts."""
    mock_site_model = mocker.patch(PATCH_SITEMODEL)
    instance = mock_site_model.return_value
    instance.get_list_of_array_layouts.return_value = ["LST", "MST", "SST"]

    def mock_side_effect(name):
        if name == "LST":
            return ["tel1", "tel2"]
        if name == "MST":
            return ["tel3", "tel4"]
        return ["tel5", "tel6"]

    instance.get_array_elements_for_layout.side_effect = mock_side_effect

    layouts = ["all"]
    site = "South"
    model_version = "7.0.0"

    result = array_layout_utils.get_array_elements_from_db_for_layouts(layouts, site, model_version)

    assert result == {
        "LST": ["tel1", "tel2"],
        "MST": ["tel3", "tel4"],
        "SST": ["tel5", "tel6"],
    }
    instance.get_list_of_array_layouts.assert_called_once()
    assert instance.get_array_elements_for_layout.call_count == 3
    instance.get_array_elements_for_layout.assert_any_call("LST")
    instance.get_array_elements_for_layout.assert_any_call("MST")
    instance.get_array_elements_for_layout.assert_any_call("SST")


def test_get_array_elements_from_db_for_layouts_single_layout(mocker):
    """Test get_array_elements_from_db_for_layouts with a single layout."""
    mock_site_model = mocker.patch(PATCH_SITEMODEL)
    instance = mock_site_model.return_value
    instance.get_array_elements_for_layout.return_value = ["tel_a", "tel_b", "tel_c"]

    layouts = ["CustomLayout"]
    site = "North"
    model_version = "6.1.0"

    result = array_layout_utils.get_array_elements_from_db_for_layouts(layouts, site, model_version)

    assert result == {"CustomLayout": ["tel_a", "tel_b", "tel_c"]}
    mock_site_model.assert_called_once_with(site=site, model_version=model_version)
    instance.get_array_elements_for_layout.assert_called_once_with("CustomLayout")


def test_get_array_elements_from_db_for_layouts_empty_layout_list(mocker):
    """Test get_array_elements_from_db_for_layouts with empty layout list."""
    mock_site_model = mocker.patch(PATCH_SITEMODEL)
    instance = mock_site_model.return_value

    layouts = []
    site = "North"
    model_version = "6.0.0"

    result = array_layout_utils.get_array_elements_from_db_for_layouts(layouts, site, model_version)

    assert result == {}
    instance.get_array_elements_for_layout.assert_not_called()


def test_get_array_elements_from_db_for_layouts_all_with_no_layouts(mocker):
    """Test get_array_elements_from_db_for_layouts with 'all' when no layouts exist."""
    mock_site_model = mocker.patch(PATCH_SITEMODEL)
    instance = mock_site_model.return_value
    instance.get_list_of_array_layouts.return_value = []

    layouts = ["all"]
    site = "South"
    model_version = "7.0.0"

    result = array_layout_utils.get_array_elements_from_db_for_layouts(layouts, site, model_version)

    assert result == {}
    instance.get_list_of_array_layouts.assert_called_once()
    instance.get_array_elements_for_layout.assert_not_called()
