#!/usr/bin/python3

from unittest.mock import MagicMock, patch

import pytest

from simtools.layout import array_layout_sources

PATCH_ASCII_COLLECT_FILE = (
    "simtools.layout.array_layout_sources.ascii_handler.collect_data_from_file"
)

PATCH_SITEMODEL = "simtools.layout.array_layout_sources.SiteModel"


@pytest.fixture
def mock_read_table_from_file():
    return "simtools.layout.array_layout_sources.data_reader.read_table_from_file"


@pytest.fixture
def mock_array_model():
    with patch("simtools.layout.array_layout_sources.ArrayModel") as mock:
        yield mock


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

    results = array_layout_sources.get_array_layouts_from_parameter_file(
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
            ignore_software_version=False,
            model_reader=None,
            context=None,
        ),
        mocker.call(
            model_version=model_version,
            site="north",
            array_elements=None,
            layout_name="array2",
            ignore_software_version=False,
            model_reader=None,
            context=None,
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
        array_layout_sources.get_array_layouts_from_parameter_file("test_file.json", "6.0.0")


def test_get_array_layouts_from_parameter_file_filtered_by_name(mocker, mock_array_model):
    model_version = "6.0.0"
    fake_data = {
        "value": [
            {"name": "array1"},
            {"name": "array2"},
            {"name": "array3"},
        ],
        "site": "north",
    }
    mocker.patch(PATCH_ASCII_COLLECT_FILE, return_value=fake_data)
    fake_table = ["tel1", "tel2"]
    instance = mock_array_model.return_value
    instance.export_array_elements_as_table.return_value = fake_table

    results = array_layout_sources.get_array_layouts_from_parameter_file(
        "test_file.json", model_version, array_layout_name=["array1", "array3"]
    )

    assert len(results) == 2
    returned_names = {layout["name"] for layout in results}
    assert returned_names == {"array1", "array3"}
    assert "array2" not in returned_names


def test_get_array_layouts_from_model_repository_with_layout_name(mock_array_model):
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
    result = array_layout_sources.get_array_layouts_from_model_repository(
        layout_name, site, model_version
    )

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
        ignore_software_version=False,
        model_reader=None,
        context=None,
    )
    instance.export_array_elements_as_table.assert_called_once_with(coordinate_system="ground")
    assert result == expected


def test_get_array_layouts_from_model_repository_without_layout_name(mocker, mock_array_model):
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
    result = array_layout_sources.get_array_layouts_from_model_repository(
        layout_name, site, model_version
    )

    # Expected: a list with two dicts.
    expected = [
        {"name": "layout1", "site": site, "array_elements": fake_table1},
        {"name": "layout2", "site": site, "array_elements": fake_table2},
    ]

    # Assert that SiteModel was correctly used.
    mock_site_model.assert_called_once_with(
        site=site,
        model_version=model_version,
        ignore_software_version=False,
        model_reader=None,
        context=None,
    )
    instance_site.get_list_of_array_layouts.assert_called_once()

    # Assert that ArrayModel was called for each layout returned by SiteModel.
    calls = [
        mocker.call(
            model_version=model_version,
            site=site,
            array_elements=None,
            layout_name="layout1",
            ignore_software_version=False,
            model_reader=None,
            context=None,
        ),
        mocker.call(
            model_version=model_version,
            site=site,
            array_elements=None,
            layout_name="layout2",
            ignore_software_version=False,
            model_reader=None,
            context=None,
        ),
    ]
    mock_array_model.assert_has_calls(calls, any_order=True)

    # Assert the result matches the expected output.
    assert result == expected


def test_get_array_layouts_using_telescope_lists_from_model_repository_with_site(
    mocker, mock_array_model
):
    telescope_lists = [["tel1", "tel2"], ["tel3", "tel4"]]
    site = "North"
    fake_table = ["fake", "elements"]

    # Patch ArrayModel to return a fake table via export_array_elements_as_table.
    instance = MagicMock()
    instance.export_array_elements_as_table.return_value = fake_table
    mock_array_model.return_value = instance

    results = array_layout_sources.get_array_layouts_using_telescope_lists_from_model_repository(
        telescope_lists, site, "6.0.0", coordinate_system="ground"
    )

    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert result["name"] == "list"
        assert result["site"] == site
        assert result["array_elements"] == fake_table

    assert mock_array_model.call_count == 2


def test_get_array_layouts_using_telescope_lists_from_model_repository_without_site_single(
    mocker, mock_array_model
):
    # Case where site is None and all telescope list elements originate from the same site.
    telescope_lists = [["N_tel1", "N_tel2"]]
    site = None
    fake_table = ["fake", "elements"]

    # Patch names.get_site_from_array_element_name to always return 'north'.
    mock_names = mocker.patch("simtools.layout.array_layout_sources.names")
    mock_names.get_site_from_array_element_name.return_value = "north"

    instance = MagicMock()
    instance.export_array_elements_as_table.return_value = fake_table
    mock_array_model.return_value = instance

    results = array_layout_sources.get_array_layouts_using_telescope_lists_from_model_repository(
        telescope_lists, site, "6.1.0", coordinate_system="ground"
    )

    assert isinstance(results, list)
    assert len(results) == 1
    result = results[0]
    assert result["name"] == "list"
    assert result["site"] == "north"
    assert result["array_elements"] == fake_table


def test_get_array_layouts_using_telescope_lists_from_model_repository_without_site_multiple_error(
    mocker,
):
    # Case where site is None and telescope list elements come from different sites.
    telescope_lists = [["N_tel1", "S_tel1"]]

    def fake_get_site(name):
        return "North" if "N" in name else "South"

    mock_names = mocker.patch("simtools.layout.array_layout_sources.names")
    mock_names.get_site_from_array_element_name.side_effect = fake_get_site

    with pytest.raises(ValueError, match="Telescope list contains elements from multiple sites:"):
        array_layout_sources.get_array_layouts_using_telescope_lists_from_model_repository(
            telescope_lists, None, "6.2.0", coordinate_system="ground"
        )


def test_get_array_layouts_from_file_single_string(mocker, mock_read_table_from_file):
    fake_table = ["dummy_table"]
    mocker.patch(mock_read_table_from_file, return_value=fake_table)
    file_path = "dummy_file.txt"
    layouts = array_layout_sources.get_array_layouts_from_file(file_path)
    assert len(layouts) == 1
    expected_name = "dummy_file"  # from "dummy_file.txt"
    assert layouts[0]["name"] == expected_name
    assert layouts[0]["array_elements"] == fake_table


def test_get_array_layouts_from_file_list(mocker, mock_read_table_from_file):
    fake_table1 = ["table1"]
    fake_table2 = ["table2"]
    mocker.patch(mock_read_table_from_file, side_effect=[fake_table1, fake_table2])
    file_paths = ["file1.csv", "file2.csv"]
    layouts = array_layout_sources.get_array_layouts_from_file(file_paths)
    assert len(layouts) == 2
    assert layouts[0]["name"] == "file1"  # from "file1.csv"
    assert layouts[1]["name"] == "file2"  # from "file2.csv"
    assert layouts[0]["array_elements"] == fake_table1
    assert layouts[1]["array_elements"] == fake_table2


def test_read_array_layouts_from_repository_specific_layouts(mocker):
    mock_site_model = mocker.patch(PATCH_SITEMODEL)
    instance = mock_site_model.return_value
    instance.get_array_elements_for_layout.side_effect = lambda name: (
        [1, 2] if name == "LST" else [3, 4]
    )

    layouts = ["LST", "MST"]
    site = "North"
    model_version = "v1.0.0"

    result = array_layout_sources.get_array_elements_from_model_repository(
        layouts, site, model_version
    )

    assert result == {"LST": [1, 2], "MST": [3, 4]}
    mock_site_model.assert_called_once_with(
        site=site, model_version=model_version, model_reader=None, context=None
    )
    assert instance.get_array_elements_for_layout.call_count == 2
    instance.get_array_elements_for_layout.assert_any_call("LST")
    instance.get_array_elements_for_layout.assert_any_call("MST")


def test_read_array_layouts_from_repository_all_layouts(mocker):
    mock_site_model = mocker.patch("simtools.layout.array_layout_sources.SiteModel")
    instance = mock_site_model.return_value
    instance.get_list_of_array_layouts.return_value = ["LST", "MST"]
    instance.get_array_elements_for_layout.side_effect = lambda name: (
        [10, 20] if name == "LST" else [30, 40]
    )

    layouts = ["all"]
    site = "South"
    model_version = "v2.0.0"

    result = array_layout_sources.get_array_elements_from_model_repository(
        layouts, site, model_version
    )

    assert result == {"LST": [10, 20], "MST": [30, 40]}
    instance.get_list_of_array_layouts.assert_called_once()
    assert instance.get_array_elements_for_layout.call_count == 2
    instance.get_array_elements_for_layout.assert_any_call("LST")
    instance.get_array_elements_for_layout.assert_any_call("MST")
