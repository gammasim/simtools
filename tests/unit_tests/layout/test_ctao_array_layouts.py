#!/usr/bin/python3

from unittest.mock import Mock, patch

import pytest

import simtools.layout.ctao_array_layouts as cta_array_layouts
from simtools.layout.ctao_array_layouts import merge_array_layouts, write_array_layouts


@pytest.fixture
def test_output():
    return "test_output.json"


@pytest.fixture
def mock_io_handler(test_output):
    with patch("simtools.layout.ctao_array_layouts.io_handler.IOHandler") as mock:
        instance = Mock()
        instance.get_output_file.return_value = test_output
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_model_data_writer():
    with patch("simtools.layout.ctao_array_layouts.ModelDataWriter") as mock:
        yield mock


@pytest.fixture
def mock_metadata_collector():
    with patch("simtools.layout.ctao_array_layouts.MetadataCollector") as mock:
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


def test_get_array_element_name():
    """Test getting array element name from common identifier."""
    # Test normal case
    array_element_ids = {
        "array_elements": [{"id": "T1", "name": "telescope1"}, {"id": "T2", "name": "telescope2"}]
    }

    assert cta_array_layouts._get_array_element_name("T1", array_element_ids) == "telescope1"
    assert cta_array_layouts._get_array_element_name("T2", array_element_ids) == "telescope2"

    # Test non-existent id
    assert cta_array_layouts._get_array_element_name("T3", array_element_ids) is None

    # Test empty array elements
    empty_elements = {"array_elements": []}
    assert cta_array_layouts._get_array_element_name("T1", empty_elements) is None

    # Test missing array_elements key
    empty_dict = {}
    assert cta_array_layouts._get_array_element_name("T1", empty_dict) is None


@patch("simtools.layout.ctao_array_layouts.names")
def test_get_layouts_per_site(mock_names):
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
    layouts = cta_array_layouts._get_layouts_per_site(site, sub_arrays, array_element_ids)

    # Assert results
    assert len(layouts) == 1
    assert layouts[0]["name"] == "array1"
    assert layouts[0]["elements"] == ["N_tel1", "N_tel2"]

    # Test empty subarrays
    layouts = cta_array_layouts._get_layouts_per_site(site, {"subarrays": []}, array_element_ids)
    assert len(layouts) == 0

    # Test missing keys
    layouts = cta_array_layouts._get_layouts_per_site(site, {}, array_element_ids)
    assert len(layouts) == 0

    # Test array with no matching elements
    site = "south"
    layouts = cta_array_layouts._get_layouts_per_site(
        site,
        {"subarrays": [{"name": "array1", "array_element_ids": ["T1", "T2"]}]},
        array_element_ids,
    )
    assert len(layouts) == 0


def test_retrieve_array_layouts_from_url():
    """Test retrieving array layouts from URL."""
    with patch("simtools.layout.ctao_array_layouts.gen") as mock_gen:
        mock_gen.is_url.return_value = True

        cta_array_layouts.retrieve_array_layouts(
            site="north", repository_url="https://test.com", branch_name="test-branch"
        )

        mock_gen.is_url.assert_called_once_with("https://test.com")
        mock_gen.collect_data_from_http.assert_called_with(
            url="https://test.com/test-branch/subarray-ids.json"
        )


def test_retrieve_array_layouts_from_file(test_path):
    """Test retrieving array layouts from local file."""
    with patch("simtools.layout.ctao_array_layouts.gen") as mock_gen:
        mock_gen.is_url.return_value = False

        cta_array_layouts.retrieve_array_layouts(
            site="north", repository_url=test_path, branch_name="test-branch"
        )

        mock_gen.is_url.assert_called_once_with(test_path)
        mock_gen.collect_data_from_file.assert_called()
