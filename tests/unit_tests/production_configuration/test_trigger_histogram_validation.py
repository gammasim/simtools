import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from astropy.table import Table

from simtools.io import table_handler
from simtools.production_configuration import trigger_histograms
from simtools.production_configuration.production_file_selection import ProductionManifest

_DISCOVER_PRODUCT_MANIFESTS = "discover_product_manifests"
_FILES_KEY = "files"
_INPUTS = "inputs"
_OUTPUTS = "outputs"
_SELECT_FILE_GROUPS = "select_file_groups"

_EXTNAME = "EXTNAME"
_ANGULAR_DISTANCE_BIN_WIDTH = "angular_distance_bin_width"
_CORE_DISTANCE_BIN_WIDTH = "core_distance_bin_width"
_APPLICATION_KEY = "application"
_CONFIGURATION_KEY = "configuration"
_COUNT = "count"
_ENERGY_MIN = "energy_min"
_EVENT_FILE = "events.hdf5"
_GROUPS_KEY = "groups"
_HISTOGRAM_SETTINGS_KEY = "histogram_settings"
_INPUT_FILES_KEY = "input_files"
_INPUT_RUN_NUMBERS_KEY = "input_run_numbers"
_OUTPUT_PATH_KEY = "output_path"
_PRODUCT_FILE = "product.hdf5"
_PRODUCTION = "production"
_PRODUCTION_PATH_KEY = "production_path"
_REDUCED_EVENT_DATA = "reduced_event_data"
_HISTOGRAM_APPLICATION = "simtools-write-trigger-histograms"
_TRIGGER_HISTOGRAMS = "trigger_histograms"
_VALUE = "value"


@pytest.fixture
def histogram_product(tmp_test_directory):
    root = Path(tmp_test_directory)
    source = root / _EVENT_FILE
    source.write_bytes(b"events")
    output = root / _PRODUCT_FILE
    tables = [
        Table({_COUNT: [1]}, meta={_EXTNAME: name})
        for name in (
            trigger_histograms.TRIGGER_HISTOGRAM_METADATA_TABLE,
            trigger_histograms.TRIGGER_HISTOGRAM_BINS_TABLE,
        )
    ]
    table_handler.write_tables(tables, output, file_type="HDF5")
    group = SimpleNamespace(configuration={_ENERGY_MIN: 30}, run_numbers=[1], file_paths=[source])
    product = ProductionManifest(
        root / "product.yml",
        {
            _CONFIGURATION_KEY: group.configuration,
            _INPUT_RUN_NUMBERS_KEY: [1],
            _INPUT_FILES_KEY: {_REDUCED_EVENT_DATA: [_EVENT_FILE]},
            _FILES_KEY: {_TRIGGER_HISTOGRAMS: [_PRODUCT_FILE]},
            _HISTOGRAM_SETTINGS_KEY: {_ANGULAR_DISTANCE_BIN_WIDTH: {_VALUE: 0.5, "unit": "deg"}},
        },
    )
    config = {_ANGULAR_DISTANCE_BIN_WIDTH: "0.5 deg"}
    return group, product, config, source, output


def test_derived_group_checks_readability_runs_settings_and_freshness(histogram_product):
    group, product, config, _source, output = histogram_product
    trigger_histograms._validate_derived_group(group, product, config)
    with pytest.raises(ValueError, match="setting changed"):
        trigger_histograms._validate_derived_group(
            group, product, {_ANGULAR_DISTANCE_BIN_WIDTH: "1 deg"}
        )
    product.data[_INPUT_RUN_NUMBERS_KEY] = [1, 2]
    with pytest.raises(ValueError, match="run coverage"):
        trigger_histograms._validate_derived_group(group, product, config)
    product.data[_INPUT_RUN_NUMBERS_KEY] = [1]
    output.write_bytes(b"not HDF5")
    with pytest.raises(ValueError, match="Unreadable trigger histogram"):
        trigger_histograms._validate_derived_group(group, product, config)


def test_derived_group_checks_settings_that_use_application_defaults(histogram_product):
    group, product, config, _source, _output = histogram_product
    product.data[_HISTOGRAM_SETTINGS_KEY][_CORE_DISTANCE_BIN_WIDTH] = {
        _VALUE: 10,
        "unit": "m",
    }

    with pytest.raises(ValueError, match="setting changed"):
        trigger_histograms._validate_derived_group(group, product, config)


def test_derived_coverage_rejects_a_partial_configuration_set(mocker, tmp_test_directory):
    mocker.patch.object(
        trigger_histograms,
        _SELECT_FILE_GROUPS,
        return_value={
            _GROUPS_KEY: [
                SimpleNamespace(configuration={_ENERGY_MIN: 30}),
                SimpleNamespace(configuration={_ENERGY_MIN: 3000}),
            ],
        },
    )
    mocker.patch.object(
        trigger_histograms,
        _DISCOVER_PRODUCT_MANIFESTS,
        return_value=[
            SimpleNamespace(data={_CONFIGURATION_KEY: {_ENERGY_MIN: 30}}),
        ],
    )
    with pytest.raises(ValueError, match="coverage differs"):
        trigger_histograms.validate_derived_products(
            {
                _PRODUCTION_PATH_KEY: str(tmp_test_directory),
                _OUTPUT_PATH_KEY: str(tmp_test_directory),
            }
        )


@pytest.mark.parametrize(_APPLICATION_KEY, [_HISTOGRAM_APPLICATION, "other-application"])
def test_product_validation_is_selected_by_application(mocker, application):
    validate = mocker.patch.object(trigger_histograms, "validate_derived_products")
    configuration = {
        _PRODUCTION_PATH_KEY: _PRODUCTION,
        _OUTPUT_PATH_KEY: _PRODUCTION,
    }
    trigger_histograms.validate_derived_workflow(
        [{_APPLICATION_KEY: application, _CONFIGURATION_KEY: configuration}]
    )
    if application == _HISTOGRAM_APPLICATION:
        validate.assert_called_once_with(configuration)
    else:
        validate.assert_not_called()


def test_product_validation_skips_event_data_application(mocker):
    validate = mocker.patch.object(trigger_histograms, "validate_derived_products")
    trigger_histograms.validate_derived_workflow(
        [
            {
                _APPLICATION_KEY: _HISTOGRAM_APPLICATION,
                _CONFIGURATION_KEY: {"event_data_directory": _PRODUCTION},
            }
        ]
    )
    validate.assert_not_called()


@pytest.mark.parametrize("duplicate", [False, True])
def test_derived_products_validate_complete_groups_and_reject_duplicates(
    histogram_product, mocker, duplicate
):
    group, product, config, _source, output = histogram_product
    mocker.patch.object(
        trigger_histograms, _SELECT_FILE_GROUPS, return_value={_GROUPS_KEY: [group]}
    )
    mocker.patch.object(
        trigger_histograms,
        _DISCOVER_PRODUCT_MANIFESTS,
        return_value=[product, product] if duplicate else [product],
    )
    configuration = config | {
        _PRODUCTION_PATH_KEY: str(output.parent),
        _OUTPUT_PATH_KEY: str(output.parent),
    }
    if duplicate:
        with pytest.raises(ValueError, match="Duplicate derived configuration"):
            trigger_histograms.validate_derived_products(configuration)
    else:
        trigger_histograms.validate_derived_products(configuration)


@pytest.mark.parametrize(
    ("change", "message"),
    [(_INPUTS, "input files differ"), (_OUTPUTS, "one histogram"), ("freshness", "inputs changed")],
)
def test_derived_products_reject_changed_inputs_and_output_coverage(
    histogram_product, change, message
):
    group, product, config, source, output = histogram_product
    if change == _INPUTS:
        product.data[_INPUT_FILES_KEY][_REDUCED_EVENT_DATA] = ["other-events.hdf5"]
    elif change == _OUTPUTS:
        product.data[_FILES_KEY][_TRIGGER_HISTOGRAMS].append("other-product.hdf5")
    else:
        newer = output.stat().st_mtime_ns + 1_000_000
        os.utime(source, ns=(newer, newer))
    with pytest.raises(ValueError, match=message):
        trigger_histograms._validate_derived_group(group, product, config)


def test_derived_products_reject_readable_but_empty_histograms(histogram_product):
    group, product, config, _source, output = histogram_product
    output.unlink()
    table_handler.write_tables(
        [
            Table({_COUNT: []}, meta={_EXTNAME: name})
            for name in (
                trigger_histograms.TRIGGER_HISTOGRAM_METADATA_TABLE,
                trigger_histograms.TRIGGER_HISTOGRAM_BINS_TABLE,
            )
        ],
        output,
        file_type="HDF5",
    )
    with pytest.raises(ValueError, match="Empty trigger histogram"):
        trigger_histograms._validate_derived_group(group, product, config)


def test_derived_products_reject_missing_histogram_table(histogram_product):
    group, product, config, _source, output = histogram_product
    output.unlink()
    table_handler.write_tables(
        [
            Table(
                {_COUNT: [1]},
                meta={"EXTNAME": trigger_histograms.TRIGGER_HISTOGRAM_METADATA_TABLE},
            )
        ],
        output,
        file_type="HDF5",
    )

    with pytest.raises(ValueError, match="Unreadable trigger histogram"):
        trigger_histograms._validate_derived_group(group, product, config)
