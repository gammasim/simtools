#!/usr/bin/python3

from unittest.mock import Mock, patch

import pytest

import simtools.layout.array_layout_utils as cta_array_layouts
from simtools.layout.array_layout_utils import merge_array_layouts, write_array_layouts


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
    with patch("simtools.layout.array_layout_utils.gen") as mock_gen:
        mock_gen.is_url.return_value = True

        cta_array_layouts.retrieve_ctao_array_layouts(
            site="north", repository_url="https://test.com", branch_name="test-branch"
        )

        mock_gen.is_url.assert_called_once_with("https://test.com")
        mock_gen.collect_data_from_http.assert_called_with(
            url="https://test.com/test-branch/subarray-ids.json"
        )


def test_retrieve_ctao_array_layouts_from_file(test_path):
    """Test retrieving array layouts from local file."""
    with patch("simtools.layout.array_layout_utils.gen") as mock_gen:
        mock_gen.is_url.return_value = False

        cta_array_layouts.retrieve_ctao_array_layouts(
            site="north", repository_url=test_path, branch_name="test-branch"
        )

        mock_gen.is_url.assert_called_once_with(test_path)
        mock_gen.collect_data_from_file.assert_called()


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
