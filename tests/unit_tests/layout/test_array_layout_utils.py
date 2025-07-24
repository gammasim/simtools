#!/usr/bin/python3

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

import simtools.layout.array_layout_utils as cta_array_layouts
from simtools.layout.array_layout_utils import (
    get_array_elements_from_db_for_layouts,
    get_array_layouts_from_file,
    get_array_layouts_from_parameter_file,
    merge_array_layouts,
    write_array_layouts,
)

# Constants for patch paths
PATCH_ASCII_COLLECT_FILE = "simtools.layout.array_layout_utils.ascii_handler.collect_data_from_file"


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
        "site": "north",
        "output_path": test_path,
        "use_plain_output_path": True,
        "updated_parameter_version": "v1",
    }
    db_config = {"test": "config"}

    write_array_layouts(array_layouts, args_dict, db_config)

    mock_io_handler.return_value.set_paths.assert_called_once_with(
        output_path=test_path, use_plain_output_path=True
    )
    mock_io_handler.return_value.get_output_file.assert_called_once_with("array-layouts-v1.json")

    mock_model_data_writer.dump_model_parameter.assert_called_once_with(
        parameter_name="array_layouts",
        value=array_layouts["value"],
        instrument="north",
        parameter_version="v1",
        output_file=test_output,
        use_plain_output_path=True,
        db_config={"test": "config"},
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
    merged = merge_array_layouts(layouts_1, layouts_2)

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

    assert cta_array_layouts._get_ctao_array_element_name("T1", array_element_ids) == "telescope1"
    assert cta_array_layouts._get_ctao_array_element_name("T2", array_element_ids) == "telescope2"

    # Test non-existent id
    assert cta_array_layouts._get_ctao_array_element_name("T3", array_element_ids) is None

    # Test empty array elements
    empty_elements = {"array_elements": []}
    assert cta_array_layouts._get_ctao_array_element_name("T1", empty_elements) is None

    # Test missing array_elements key
    empty_dict = {}
    assert cta_array_layouts._get_ctao_array_element_name("T1", empty_dict) is None


@patch("simtools.layout.array_layout_utils.names")
def test_get_ctao_layouts_per_site(mock_names):
    """Test getting array layouts per site."""
    # Mock site determination function
    mock_names.get_site_from_array_element_name.side_effect = (
        lambda x: "north" if "N" in x else "south"
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
    layouts = cta_array_layouts._get_ctao_layouts_per_site(site, sub_arrays, array_element_ids)

    # Assert results
    assert len(layouts) == 1
    assert layouts[0]["name"] == "array1"
    assert layouts[0]["elements"] == ["N_tel1", "N_tel2"]

    # Test empty subarrays
    layouts = cta_array_layouts._get_ctao_layouts_per_site(
        site, {"subarrays": []}, array_element_ids
    )
    assert len(layouts) == 0

    # Test missing keys
    layouts = cta_array_layouts._get_ctao_layouts_per_site(site, {}, array_element_ids)
    assert len(layouts) == 0

    # Test array with no matching elements
    site = "south"
    layouts = cta_array_layouts._get_ctao_layouts_per_site(
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

        cta_array_layouts.retrieve_ctao_array_layouts(
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

        cta_array_layouts.retrieve_ctao_array_layouts(
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

    result = cta_array_layouts.validate_array_layouts_with_db(production_table, array_layouts)
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

    with pytest.raises(ValueError, match="Invalid array elements found: \\['tel3', 'tel4'\\]"):
        cta_array_layouts.validate_array_layouts_with_db(production_table, array_layouts)


def test_validate_array_layouts_with_db_empty_production_table():
    """Test validation with empty production table."""
    production_table = {"parameters": {}}

    array_layouts = {
        "value": [
            {"name": "array1", "elements": ["tel1"]},
        ]
    }

    with pytest.raises(ValueError, match="Invalid array elements found: \\['tel1'\\]"):
        cta_array_layouts.validate_array_layouts_with_db(production_table, array_layouts)


def test_validate_array_layouts_with_db_empty_array_layouts():
    """Test validation with empty array layouts."""
    production_table = {"parameters": {"tel1": {}, "tel2": {}}}

    array_layouts = {"value": []}

    result = cta_array_layouts.validate_array_layouts_with_db(production_table, array_layouts)
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

    with pytest.raises(ValueError, match="Invalid array elements found: \\['tel1'\\]"):
        cta_array_layouts.validate_array_layouts_with_db(production_table, array_layouts)

    # Missing value key
    production_table = {"parameters": {"tel1": {}}}
    array_layouts = {}

    result = cta_array_layouts.validate_array_layouts_with_db(production_table, array_layouts)
    assert result == array_layouts

    # Missing elements key in layout
    array_layouts = {
        "value": [
            {"name": "array1"},  # No elements key
        ]
    }

    result = cta_array_layouts.validate_array_layouts_with_db(production_table, array_layouts)
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

    with pytest.raises(ValueError, match="Invalid array elements found: \\['tel3'\\]"):
        cta_array_layouts.validate_array_layouts_with_db(production_table, array_layouts)


def test_get_array_layouts_from_parameter_file_valid(mocker, mock_array_model):
    model_version = "6.0.0"
    db_config = {"dummy": "config"}
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

    results = get_array_layouts_from_parameter_file("test_file.json", model_version, db_config)

    assert isinstance(results, list)
    assert len(results) == 2
    for layout in results:
        assert layout["site"] == "north"
        assert layout["array_elements"] == fake_table
        assert layout["name"] in ["array1", "array2"]

    expected_calls = [
        mocker.call(
            mongo_db_config=db_config,
            model_version=model_version,
            site="north",
            array_elements=None,
            layout_name="array1",
        ),
        mocker.call(
            mongo_db_config=db_config,
            model_version=model_version,
            site="north",
            array_elements=None,
            layout_name="array2",
        ),
    ]
    mock_array_model.assert_has_calls(expected_calls, any_order=True)


def test_get_array_layouts_from_parameter_file_missing_value_key(mocker):
    db_config = {"dummy": "config"}
    fake_data = {
        "site": "north",
    }
    # Patch ascii_handler.collect_data_from_file to return fake_data without the "value" key.
    mocker.patch(
        PATCH_ASCII_COLLECT_FILE,
        return_value=fake_data,
    )

    with pytest.raises(ValueError, match="Missing 'value' key in layout file."):
        get_array_layouts_from_parameter_file("test_file.json", "6.0.0", db_config)


def test_get_array_layouts_from_db_with_layout_name(mock_array_model):
    # Test when a specific layout_name is provided.
    layout_name = "layout_test"
    site = "North"
    model_version = "6.0.0"
    db_config = {"dummy": "config"}
    fake_table = ["tel1", "tel2"]

    # Patch ArrayModel so that export_array_elements_as_table returns fake_table.
    instance = MagicMock()
    instance.export_array_elements_as_table.return_value = fake_table
    mock_array_model.return_value = instance

    # Call the function with layout_name provided.
    result = cta_array_layouts.get_array_layouts_from_db(
        layout_name, site, model_version, db_config
    )

    # Expected: a list with one dict corresponding to the provided layout_name.
    expected = {
        "name": layout_name,
        "site": site,
        "array_elements": fake_table,
    }

    # Assert that ArrayModel was initialized with the correct parameters.
    mock_array_model.assert_called_once_with(
        mongo_db_config=db_config,
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
    db_config = {"dummy": "config"}
    # Fake layout names returned by SiteModel.
    fake_layout_names = ["layout1", "layout2"]

    # Patch SiteModel so that get_list_of_array_layouts returns our fake_layout_names.
    mock_site_model = mocker.patch("simtools.layout.array_layout_utils.SiteModel")
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
    result = cta_array_layouts.get_array_layouts_from_db(
        layout_name, site, model_version, db_config
    )

    # Expected: a list with two dicts.
    expected = [
        {"name": "layout1", "site": site, "array_elements": fake_table1},
        {"name": "layout2", "site": site, "array_elements": fake_table2},
    ]

    # Assert that SiteModel was correctly used.
    mock_site_model.assert_called_once_with(
        site=site, model_version=model_version, mongo_db_config=db_config
    )
    instance_site.get_list_of_array_layouts.assert_called_once()

    # Assert that ArrayModel was called for each layout returned by SiteModel.
    calls = [
        mocker.call(
            mongo_db_config=db_config,
            model_version=model_version,
            site=site,
            array_elements=None,
            layout_name="layout1",
        ),
        mocker.call(
            mongo_db_config=db_config,
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
    db_config = {"config": "dummy"}
    fake_table = ["fake", "elements"]

    # Patch ArrayModel to return a fake table via export_array_elements_as_table.
    instance = MagicMock()
    instance.export_array_elements_as_table.return_value = fake_table
    mock_array_model.return_value = instance

    results = cta_array_layouts.get_array_layouts_using_telescope_lists_from_db(
        telescope_lists, site, "6.0.0", db_config, coordinate_system="ground"
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
    db_config = {"config": "dummy"}
    fake_table = ["fake", "elements"]

    # Patch names.get_site_from_array_element_name to always return 'north'.
    mock_names = mocker.patch("simtools.layout.array_layout_utils.names")
    mock_names.get_site_from_array_element_name.return_value = "north"

    instance = MagicMock()
    instance.export_array_elements_as_table.return_value = fake_table
    mock_array_model.return_value = instance

    results = cta_array_layouts.get_array_layouts_using_telescope_lists_from_db(
        telescope_lists, site, "6.1.0", db_config, coordinate_system="ground"
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
    db_config = {"config": "dummy"}

    def fake_get_site(name):
        return "North" if "N" in name else "South"

    mock_names = mocker.patch("simtools.layout.array_layout_utils.names")
    mock_names.get_site_from_array_element_name.side_effect = fake_get_site

    with pytest.raises(ValueError, match="Telescope list contains elements from multiple sites:"):
        cta_array_layouts.get_array_layouts_using_telescope_lists_from_db(
            telescope_lists, None, "6.2.0", db_config, coordinate_system="ground"
        )


def test_get_array_layouts_from_file_single_string(mocker, mock_read_table_from_file):
    fake_table = ["dummy_table"]
    mocker.patch(mock_read_table_from_file, return_value=fake_table)
    file_path = "dummy_file.txt"
    layouts = get_array_layouts_from_file(file_path)
    assert len(layouts) == 1
    expected_name = "dummy_file"  # from "dummy_file.txt"
    assert layouts[0]["name"] == expected_name
    assert layouts[0]["array_elements"] == fake_table


def test_get_array_layouts_from_file_single_path(mocker, mock_read_table_from_file):
    fake_table = ["path_table"]
    mocker.patch(mock_read_table_from_file, return_value=fake_table)
    file_path = Path("example_file.dat")
    layouts = get_array_layouts_from_file(file_path)
    assert len(layouts) == 1
    expected_name = "example_file"  # from "example_file.dat"
    assert layouts[0]["name"] == expected_name
    assert layouts[0]["array_elements"] == fake_table


def test_get_array_layouts_from_file_list(mocker, mock_read_table_from_file):
    fake_table1 = ["table1"]
    fake_table2 = ["table2"]
    mocker.patch(mock_read_table_from_file, side_effect=[fake_table1, fake_table2])
    file_paths = ["file1.csv", "file2.csv"]
    layouts = get_array_layouts_from_file(file_paths)
    assert len(layouts) == 2
    assert layouts[0]["name"] == "file1"  # from "file1.csv"
    assert layouts[1]["name"] == "file2"  # from "file2.csv"
    assert layouts[0]["array_elements"] == fake_table1
    assert layouts[1]["array_elements"] == fake_table2


def test_get_array_layout_dict_with_layout_name(mock_array_model):
    """Test _get_array_layout_dict with a layout name provided."""
    # Setup test data
    db_config = {"db": "config"}
    model_version = "6.0.0"
    site = "north"
    layout_name = "test_layout"
    fake_table = ["tel1", "tel2"]

    # Mock ArrayModel instance
    instance = MagicMock()
    instance.export_array_elements_as_table.return_value = fake_table
    mock_array_model.return_value = instance

    # Call function
    result = cta_array_layouts._get_array_layout_dict(
        db_config, model_version, site, None, layout_name, "ground"
    )

    # Verify ArrayModel initialization
    mock_array_model.assert_called_once_with(
        mongo_db_config=db_config,
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
    db_config = {"db": "config"}
    model_version = "6.0.0"
    site = "south"
    telescope_list = ["tel1", "tel2", "tel3"]
    fake_table = ["tel_data1", "tel_data2", "tel_data3"]

    # Mock ArrayModel instance
    instance = MagicMock()
    instance.export_array_elements_as_table.return_value = fake_table
    mock_array_model.return_value = instance

    # Call function
    result = cta_array_layouts._get_array_layout_dict(
        db_config, model_version, site, telescope_list, None, "ground"
    )

    # Verify ArrayModel initialization
    mock_array_model.assert_called_once_with(
        mongo_db_config=db_config,
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
    mock_site_model = mocker.patch("simtools.layout.array_layout_utils.SiteModel")
    instance = mock_site_model.return_value
    instance.get_array_elements_for_layout.side_effect = (
        lambda name: [1, 2] if name == "LST" else [3, 4]
    )

    layouts = ["LST", "MST"]
    site = "North"
    model_version = "v1.0.0"
    db_config = {"host": "localhost"}

    result = get_array_elements_from_db_for_layouts(layouts, site, model_version, db_config)

    assert result == {"LST": [1, 2], "MST": [3, 4]}
    mock_site_model.assert_called_once_with(
        site=site, model_version=model_version, mongo_db_config=db_config
    )
    assert instance.get_array_elements_for_layout.call_count == 2
    instance.get_array_elements_for_layout.assert_any_call("LST")
    instance.get_array_elements_for_layout.assert_any_call("MST")


def test_read_array_layouts_from_db_all_layouts(mocker):
    """Test _read_array_layouts_from_db with 'all' layouts."""
    mock_site_model = mocker.patch("simtools.layout.array_layout_utils.SiteModel")
    instance = mock_site_model.return_value
    instance.get_list_of_array_layouts.return_value = ["LST", "MST"]
    instance.get_array_elements_for_layout.side_effect = (
        lambda name: [10, 20] if name == "LST" else [30, 40]
    )

    layouts = ["all"]
    site = "South"
    model_version = "v2.0.0"
    db_config = {"host": "db"}

    result = get_array_elements_from_db_for_layouts(layouts, site, model_version, db_config)

    assert result == {"LST": [10, 20], "MST": [30, 40]}
    instance.get_list_of_array_layouts.assert_called_once()
    assert instance.get_array_elements_for_layout.call_count == 2
    instance.get_array_elements_for_layout.assert_any_call("LST")
    instance.get_array_elements_for_layout.assert_any_call("MST")
