"""Tests for reduced-event metadata helpers."""

import astropy.units as u
import pytest
from bson.objectid import ObjectId

from simtools.constants import METADATA_JSON_SCHEMA
from simtools.data_model import schema
from simtools.model.array_model import ArrayModel
from simtools.sim_events.metadata import (
    build_simulation_metadata,
    build_standard_metadata,
    validate_simulation_metadata,
)


def test_build_standard_metadata_keeps_required_metadata_fields():
    """Produce a complete standard metadata document for embedded HDF5 output."""
    metadata = build_standard_metadata({}, "reduced_event_data.hdf5")

    schema.validate_dict_using_schema(metadata, schema_file=METADATA_JSON_SCHEMA)
    assert "instrument" in metadata["cta"]


def test_build_standard_metadata_uses_requested_product_data_name():
    """Allow products other than reduced-event lists to identify themselves."""
    metadata = build_standard_metadata(
        {}, "trigger_histograms.hdf5", product_data_name="trigger_histograms"
    )

    assert metadata["cta"]["product"]["data"]["model"]["name"] == "trigger_histograms"


def test_build_standard_metadata_normalizes_instrument_id(mocker):
    mocker.patch(
        "simtools.sim_events.metadata.MetadataCollector",
        return_value=type(
            "Collector",
            (),
            {"get_top_level_metadata": lambda self: {"cta": {"instrument": {"ID": "CTA"}}}},
        )(),
    )

    metadata = build_standard_metadata({}, "output.hdf5")

    assert metadata["cta"]["instrument"] == {"id": "CTA"}


def test_build_simulation_metadata_marks_missing_models_unavailable():
    metadata = build_simulation_metadata([{"run_number": 12}])

    assert metadata["models"]["status"] == "unavailable"
    assert metadata["models"]["arrays"] == []
    assert metadata["inputs"]["files"][0]["run_number"] == 12


def test_build_simulation_metadata_marks_supplied_models_complete():
    metadata = build_simulation_metadata(
        [{"run_number": 12}], array_models=[{"layout_name": "alpha"}], simulation_software="corsika"
    )

    assert metadata["models"] == {
        "status": "complete",
        "arrays": [{"layout_name": "alpha"}],
    }
    assert metadata["provenance"]["simulation_software"] == {"name": "corsika"}


def test_array_model_export_keeps_telescope_context_and_parameter_records():
    array_model = ArrayModel.__new__(ArrayModel)
    array_model.model_version = "7.0.0"
    array_model.layout_name = "layout"
    array_model.array_elements = {
        "LSTN-01": {"value": [1, 2, 3] * u.m},
    }
    array_model.site_model = type(
        "Site",
        (),
        {
            "site": "North",
            "model_version": "7.0.0",
            "parameters": {"altitude": {"value": 2, "unit": "km"}},
        },
    )()
    telescope = type(
        "Telescope",
        (),
        {
            "design_model": "LST",
            "model_version": "7.0.0",
            "site": "North",
            "parameters": {"camera_name": {"value": "cam"}},
            "name": "LSTN-01",
        },
    )()
    array_model.telescope_models = {"LSTN-01": telescope}
    array_model.calibration_models = {"LSTN-01": {}}

    exported = array_model.to_simulation_metadata_dict()
    record = exported["telescopes"]["LSTN-01"]

    assert record["design_model"] == "LST"
    assert record["site_name"] == "North"
    assert record["parameters"]["camera_name"]["value"] == "cam"
    assert exported["site_model"]["site_name"] == "North"
    assert exported["site_model"]["model_version"] == "7.0.0"
    assert exported["site_model"]["parameters"]["altitude"] == {
        "value": 2.0,
        "unit": "km",
    }


def test_array_model_export_excludes_database_bookkeeping_fields():
    array_model = ArrayModel.__new__(ArrayModel)
    array_model.model_version = "7.0.0"
    array_model.layout_name = "layout"
    array_model.array_elements = {}
    database_parameter = {
        "value": [],
        "unit": None,
        "_id": ObjectId(),
        "entry_date": "database timestamp",
    }
    array_model.site_model = type(
        "Site",
        (),
        {
            "site": "North",
            "model_version": "7.0.0",
            "parameters": {"array_layouts": database_parameter},
        },
    )()
    array_model.telescope_models = {}
    array_model.calibration_models = {}

    exported = array_model.to_simulation_metadata_dict()

    assert exported["site_model"]["parameters"] == {"array_layouts": {"value": [], "unit": None}}


def test_validate_simulation_metadata_rejects_model_arrays_when_unavailable():
    metadata = build_simulation_metadata([])
    metadata["models"]["arrays"] = [{"model_version": "7.0.0"}]

    with pytest.raises(ValueError, match="must not contain model arrays"):
        validate_simulation_metadata(metadata)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "missing required field"),
        (
            {
                "schema_name": "other",
                "schema_version": "1.0.0",
                "provenance": {},
                "inputs": {},
                "models": {"status": "complete"},
            },
            "Unsupported simulation metadata schema",
        ),
        (
            {
                "schema_name": "simtools.reduced_event_list.simulation_metadata",
                "schema_version": "1.0.0",
                "provenance": {},
                "inputs": {},
                "models": {"status": "invalid"},
            },
            "models.status",
        ),
    ],
)
def test_validate_simulation_metadata_rejects_invalid_documents(metadata, message):
    with pytest.raises(ValueError, match=message):
        validate_simulation_metadata(metadata)
