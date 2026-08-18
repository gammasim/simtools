"""Shared pytest configuration."""

import os
from pathlib import Path

import pytest

SIMTOOLS_ROOT_PATH = Path(__file__).resolve().parent.parent
SIMTOOLS_TEST_PATH = (
    Path(os.environ["SIMTOOLS_TEST_PATH"]).expanduser()
    if os.environ.get("SIMTOOLS_TEST_PATH")
    else None
)

EXTERNAL_RESOURCE_TESTS = {
    "tests/unit_tests/data_model/test_data_reader.py::test_read_table_from_file_and_validate",
    "tests/unit_tests/data_model/test_metadata_collector.py::test_get_site",
    "tests/unit_tests/data_model/test_metadata_collector.py::test_read_input_metadata_from_file",
    "tests/unit_tests/data_model/test_model_data_writer.py::test_validate_and_transform",
    "tests/unit_tests/data_model/test_validate_data.py::test_validate_and_transform",
    "tests/unit_tests/data_model/test_validate_data.py::test_validate_data_file",
    "tests/unit_tests/data_model/test_validate_data.py::test_validate_parameter_and_file_name",
    "tests/unit_tests/data_model/test_validate_data.py::test_validate_data_columns",
    "tests/unit_tests/data_model/test_validate_data.py::test_validate_data_files_single_file",
    "tests/unit_tests/data_model/test_validate_data.py::test_validate_data_files_with_schema_file",
    "tests/unit_tests/io/test_ascii_handler.py::test_collect_dict_data",
    "tests/unit_tests/layout/test_array_layout_utils.py::test_write_array_elements_from_file_to_repository_utm",
    "tests/unit_tests/layout/test_array_layout_utils.py::test_write_array_elements_from_file_to_repository_ground",
    "tests/unit_tests/layout/test_array_layout_utils.py::test_write_array_elements_from_file_to_repository_error",
    "tests/unit_tests/model/test_array_model.py::test_array_model_north_from_file",
    "tests/unit_tests/production_configuration/test_corsika_limits_lookup.py::test_load_matching_lookup_arrays_filters_by_array_layout_name",
    "tests/unit_tests/production_configuration/test_corsika_limits_lookup.py::test_load_matching_lookup_arrays_raises_for_unknown_array_layout",
    "tests/unit_tests/production_configuration/test_corsika_limits_lookup.py::test_load_matching_lookup_arrays_without_layout_returns_all_rows",
    "tests/unit_tests/production_configuration/test_corsika_limits_lookup.py::test_prepare_point_interpolators_builds_interpolator_state",
    "tests/unit_tests/production_configuration/test_corsika_limits_lookup.py::test_interpolate_grid_limits_returns_requested_grid_shape",
    "tests/unit_tests/production_configuration/test_corsika_limits_lookup.py::test_interpolate_point_returns_interpolated_values",
    "tests/unit_tests/ray_tracing/test_psf_analysis.py::test_reading_simtel_file",
    "tests/unit_tests/sim_events/test_file_info.py::test_get_corsika_run_number_with_run_header",
    "tests/unit_tests/sim_events/test_file_info.py::test_get_simulated_events",
    "tests/unit_tests/sim_events/test_file_info.py::test_get_simulated_events_corsika_iact",
    "tests/unit_tests/sim_events/test_file_info.py::test_get_corsika_run_and_event_headers",
    "tests/unit_tests/sim_events/test_writer.py::test_chunked_output_matches_non_chunked_output",
    "tests/unit_tests/simtel/test_simtel_config_reader.py::test_simtel_config_reader_init",
    "tests/unit_tests/simtel/test_simtel_config_reader.py::test_get_list_of_simtel_parameters",
    "tests/unit_tests/simtel/test_simtel_config_reader.py::test_simtel_config_reader_num_gains",
    "tests/unit_tests/simtel/test_simtel_config_reader.py::test_simtel_config_reader_telescope_transmission",
    "tests/unit_tests/simtel/test_simtel_config_reader.py::test_compare_simtel_config_with_schema",
    "tests/unit_tests/simtel/test_simtel_config_reader.py::test_read_simtel_config_file",
    "tests/unit_tests/simtel/test_simtel_config_reader.py::test_get_type_and_dimension_from_simtel_cfg",
    "tests/unit_tests/simtel/test_simtel_config_reader.py::test_resolve_all_in_column",
    "tests/unit_tests/simtel/test_simtel_config_reader.py::testextract_value_from_sim_telarray_column",
    "tests/unit_tests/simtel/test_simtel_config_reader.py::test_values_match",
    "tests/unit_tests/simtel/test_simtel_config_reader.py::test_add_units",
    "tests/unit_tests/simtel/test_simtel_io_metadata.py::test_read_sim_telarray_metadata",
    "tests/unit_tests/simtel/test_simtel_io_metadata.py::test_read_sim_telarray_metadata_attribute_error",
    "tests/unit_tests/simtel/test_simtel_io_metadata.py::test_get_sim_telarray_telescope_id",
    "tests/unit_tests/simtel/test_simtel_io_metadata.py::test_get_sim_telarray_telescope_id_to_telescope_name_mapping",
    "tests/unit_tests/simtel/test_simtel_table_reader.py::test_read_simtel_data",
    "tests/unit_tests/simtel/test_simtel_table_reader.py::test_read_simtel_table_to_table",
    "tests/unit_tests/simtel/test_simtel_table_reader.py::test_read_simtel_table_for_mirror_list",
    "tests/unit_tests/simtel/test_simtel_table_reader.py::test_read_simtel_table_as_row_data_for_mirror_list",
    "tests/unit_tests/testing/test_validate_output.py::test_compare_simtel_cfg_files",
    "tests/unit_tests/visualization/test_plot_tables.py::test_read_simtel_table_data_from_file",
    "tests/unit_tests/visualization/test_plot_tables.py::test_read_table_and_normalize",
    "tests/unit_tests/visualization/test_visualize.py::test_plot_table",
}


def _versioned_test_resources_path(version):
    """Return the selected local version of the integration test resources."""
    if SIMTOOLS_TEST_PATH is None or not version:
        return None
    return SIMTOOLS_TEST_PATH / version / "integration_tests"


def _configured_test_resources_path(config):
    """Return the absolute path to the configured test resources directory."""
    configured_path = config.getoption("test_resources_path", default=None)
    path = configured_path or os.environ.get("SIMTOOLS_TEST_RESOURCES")
    version = config.getoption("simtools_tests_version", default=None) or os.environ.get(
        "SIMTOOLS_TESTS_VERSION"
    )
    path = path or _versioned_test_resources_path(version)
    path = path or SIMTOOLS_ROOT_PATH / "tests" / "unit_tests" / "resources"
    return Path(path).expanduser().resolve()


def pytest_addoption(parser):
    """Register test-resource configuration options."""
    parser.addoption(
        "--test_resources_path",
        dest="test_resources_path",
        type=Path,
        default=os.environ.get("SIMTOOLS_TEST_RESOURCES"),
        help="Full path to test resources (default: SIMTOOLS_TEST_RESOURCES).",
    )
    parser.addoption(
        "--simtools_tests_version",
        dest="simtools_tests_version",
        default=os.environ.get("SIMTOOLS_TESTS_VERSION"),
        help="Version of simtools-tests to use when no path is configured.",
    )


def pytest_configure(config):
    """Configure test resource constants before test modules are imported."""
    import simtools.constants

    test_resources_path = _configured_test_resources_path(config)
    config.option.test_resources_path = test_resources_path
    simtools.constants.TEST_RESOURCES_ROOT = test_resources_path
    simtools.constants.TEST_RESOURCES_STATIC = str(test_resources_path / "static")
    simtools.constants.TEST_RESOURCES_GENERATED = str(test_resources_path / "generated")
    simtools.constants.TEST_RESOURCES_DOWNLOADED = str(test_resources_path / "downloaded")


def pytest_collection_modifyitems(config, items):
    """Mark unit tests requiring the external resource bundle as expected failures."""
    local_resources_path = (SIMTOOLS_ROOT_PATH / "tests" / "unit_tests" / "resources").resolve()
    if _configured_test_resources_path(config) != local_resources_path:
        return

    marker = pytest.mark.xfail(
        reason="Requires external test resources from simtools-tests",
        strict=False,
    )
    for item in items:
        if item.nodeid in EXTERNAL_RESOURCE_TESTS:
            item.add_marker(marker)


@pytest.fixture(scope="session")
def test_resources_path(pytestconfig):
    """Return the absolute path to the test resources directory."""
    return _configured_test_resources_path(pytestconfig)


@pytest.fixture(scope="session")
def simtools_root_path():
    """Return the path to the simtools repository root."""
    return SIMTOOLS_ROOT_PATH
