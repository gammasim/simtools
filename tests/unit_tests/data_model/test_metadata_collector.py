#!/usr/bin/python3

import copy
import getpass
import json
import logging
import re
import uuid
from pathlib import Path

import astropy.units as u
import jsonschema
import numpy as np
import pytest
from astropy.table import Table

import simtools.data_model.metadata_collector as metadata_collector
from simtools.constants import METADATA_JSON_SCHEMA, SCHEMA_PATH
from simtools.data_model import schema
from simtools.sim_events import metadata_reader as event_metadata_reader
from simtools.simtel import metadata_reader as simtel_metadata_reader

logger = logging.getLogger()


def test_associated_data_keeps_required_data_description(mocker):
    collector = metadata_collector.MetadataCollector.__new__(metadata_collector.MetadataCollector)
    collector.observatory = "cta"
    collector.args_dict = {}
    collector._logger = logger
    data = {"category": "SIM", "level": "R1", "model": {"name": "setting", "version": "1.0.0"}}
    collector.input_metadata = [
        {"cta": {"product": {"id": "input-id", "format": "yaml", "data": data}}}
    ]
    mocker.patch.object(collector, "_append_context_note_from_value_table")
    mocker.patch.object(collector, "_fill_application_configuration")
    context = {"associated_data": []}
    collector._fill_context_meta(context)
    assert context["associated_data"] == [{"id": "input-id", "format": "yaml", "data": data}]


def test_get_data_model_schema_file_name():
    # from args_dict / command line
    args_dict = {"no_schema": "schema_file.yml"}
    _collector = metadata_collector.MetadataCollector(args_dict)
    schema_file = _collector.get_data_model_schema_file_name()
    assert schema_file is None

    args_dict = {"schema_file": str(METADATA_JSON_SCHEMA)}
    _collector = metadata_collector.MetadataCollector(args_dict, clean_meta=False)
    schema_file = _collector.get_data_model_schema_file_name()
    assert schema_file == args_dict["schema_file"]

    # from metadata
    _collector.top_level_meta["cta"]["product"]["data"]["model"]["url"] = str(
        SCHEMA_PATH / "top_level_meta.schema.yml"
    )
    schema_file = _collector.get_data_model_schema_file_name()
    # test that priority is given to args_dict (if not none)
    assert schema_file == args_dict["schema_file"]
    _collector.args_dict["schema_file"] = None
    schema_file = _collector.get_data_model_schema_file_name()
    assert schema_file == str(SCHEMA_PATH / "top_level_meta.schema.yml")

    _collector.top_level_meta["cta"]["product"]["data"]["model"].pop("url")
    schema_file = _collector.get_data_model_schema_file_name()
    assert schema_file is None

    # from data model_name
    _collector.model_parameter_name = "array_coordinates"
    schema_file = _collector.get_data_model_schema_file_name()
    assert Path(schema_file) == (
        schema.get_model_parameter_schema_file(_collector.model_parameter_name)
    )

    # from input metadata
    _collector.input_metadata = [
        {"cta": {"product": {"data": {"model": {"url": "from_input_meta"}}}}}
    ]
    _collector.model_parameter_name = None
    schema_file = _collector.get_data_model_schema_file_name()
    assert schema_file == "from_input_meta"


def test_get_top_level_metadata(args_dict_site, mocker):
    mocker.patch(
        "simtools.data_model.metadata_collector.gen.now_date_time_in_isoformat",
        side_effect=[
            "2020-01-01T00:00:00+00:00",
            "2020-01-01T00:00:00+00:00",
            "2020-01-01T00:00:00+00:00",
            "2099-01-01T00:00:00+00:00",
        ],
    )
    collector = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    assert (
        collector.top_level_meta["cta"]["activity"]["end"]
        == collector.top_level_meta["cta"]["activity"]["start"]
    )

    # no update when activity cannot be found in the metadata
    collector.observatory = "not_cta"
    top_level_meta = collector.get_top_level_metadata()
    assert top_level_meta["cta"]["activity"]["end"] == top_level_meta["cta"]["activity"]["start"]

    collector.observatory = "cta"  # back to default
    top_level_meta = collector.get_top_level_metadata()
    assert top_level_meta["cta"]["activity"]["end"] > top_level_meta["cta"]["activity"]["start"]


def test_fill_contact_meta(args_dict_site, caplog):
    contact_dict = {}
    collector = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    collector._fill_contact_meta(contact_dict)
    with caplog.at_level(logging.WARNING):
        collector._fill_contact_meta(contact_dict)
    assert "No user name provided, take user info from system level." not in caplog.text
    try:
        assert contact_dict["name"] == getpass.getuser()
    except Exception:  # pylint: disable=broad-except
        pass


def test_get_site(args_dict_site, tmp_test_directory):
    _collector_1 = metadata_collector.MetadataCollector(
        args_dict=args_dict_site,
    )
    assert _collector_1.get_site() == "South"
    assert _collector_1.get_site(from_input_meta=True) is None

    input_metadata = copy.deepcopy(
        metadata_collector.MetadataCollector(
            args_dict=args_dict_site, clean_meta=False
        ).top_level_meta
    )
    input_metadata["cta"]["instrument"]["site"] = "North"
    input_metadata["cta"]["instrument"].pop("ID", None)
    input_metadata["cta"]["instrument"]["id"] = "MSTS-07"
    metadata_file = Path(tmp_test_directory) / "north_metadata.json"
    metadata_file.write_text(json.dumps(input_metadata), encoding="utf-8")
    _collector_2 = metadata_collector.MetadataCollector(
        args_dict=args_dict_site, metadata_file_name=metadata_file
    )
    assert _collector_2.get_site(from_input_meta=True) == "North"
    assert _collector_2.get_site(from_input_meta=False) == "South"  # from args_dict


def test_read_input_metadata_from_file(
    args_dict_site, tmp_test_directory, simple_test_file, mocker
):
    metadata_1 = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    metadata_1.args_dict["input_meta"] = None

    assert metadata_1._read_input_metadata_from_file() is None

    metadata_1.args_dict["input_meta"] = "./file_does_not_exist.yml"
    with pytest.raises(
        FileNotFoundError, match=re.escape("No files found: ['./file_does_not_exist.yml']")
    ):
        metadata_1._read_input_metadata_from_file()

    metadata_1.args_dict["input_meta"] = None
    local_metadata_file = Path(tmp_test_directory) / "input_metadata.json"
    source_metadata = metadata_collector.MetadataCollector(
        args_dict=args_dict_site, clean_meta=False
    ).top_level_meta
    source_metadata["cta"]["instrument"].pop("ID", None)
    source_metadata["cta"]["instrument"]["id"] = "MSTS-07"
    local_metadata_file.write_text(json.dumps(source_metadata), encoding="utf-8")
    metadata_1.args_dict["input_meta"] = local_metadata_file
    assert len(metadata_1._read_input_metadata_from_file()) > 0

    test_dict = {
        "metadata": {"cta": {"product": {"data": {"model": {"url": "from_input_meta"}}}}},
        "METADATA": {"cta": {"product": {"data": {"model": {"url": "from_input_meta"}}}}},
    }
    with open(tmp_test_directory / "test_read_input_metadata_file.json", "w") as f:
        json.dump(test_dict, f)
    metadata_1.args_dict["input_meta"] = tmp_test_directory / "test_read_input_metadata_file.json"
    with pytest.raises(ValueError, match=r"^More than one metadata entry found in"):
        metadata_1._read_input_metadata_from_file()

    table = Table(
        {"value": [1]},
        meta={"CTA": source_metadata["cta"]},
    )
    local_ecsv_file = Path(tmp_test_directory) / "input_metadata.ecsv"
    table.write(local_ecsv_file, format="ascii.ecsv", overwrite=True)
    metadata_1.args_dict["input_meta"] = local_ecsv_file
    validate_schema = mocker.patch.object(metadata_collector.schema, "validate_dict_using_schema")
    assert len(metadata_1._read_input_metadata_from_file()) > 0
    mocker.stop(validate_schema)

    metadata_1.args_dict["input_meta"] = "tests/resources/file_not_there.ecsv"
    with pytest.raises(FileNotFoundError, match=r"^No files found:"):
        metadata_1._read_input_metadata_from_file()

    simtel_file = Path(tmp_test_directory) / "gamma.simtel.zst"
    simtel_file.write_bytes(b"")
    mocker.patch.object(
        simtel_metadata_reader,
        "read_sim_telarray_metadata",
        return_value=(
            {
                "site_config_name": "South",
                "array_config_name": "CTAO-South-Alpha",
                "simtools_simtel_tag": "v2026-01-01",
                "simtools_model_production_version": "6.0.2",
            },
            {
                1: {"optics_config_variant": "MSTS-01"},
            },
        ),
    )
    validate_simtel_metadata = mocker.patch.object(
        simtel_metadata_reader.sim_telarray_metadata,
        "validate_metadata_values",
    )
    metadata_1.args_dict["input_meta"] = simtel_file
    simtel_metadata = metadata_1._read_input_metadata_from_file()[0]
    assert validate_simtel_metadata.call_args_list == [
        mocker.call(
            {
                "site_config_name": "South",
                "array_config_name": "CTAO-South-Alpha",
                "simtools_simtel_tag": "v2026-01-01",
                "simtools_model_production_version": "6.0.2",
            }
        ),
        mocker.call({"optics_config_variant": "MSTS-01"}),
    ]
    assert simtel_metadata["cta"]["product"]["filename"] == str(simtel_file)
    assert simtel_metadata["cta"]["product"]["format"] == "simtel"
    assert simtel_metadata["cta"]["instrument"]["site"] == "South"
    assert simtel_metadata["cta"]["instrument"]["id"] == "CTAO-South-Alpha"
    assert simtel_metadata["cta"]["product"]["data"]["model"]["version"] is None
    assert (
        '"simtools_model_production_version": "6.0.2"'
        in simtel_metadata["cta"]["context"]["notes"][0]["text"]
    )
    assert simtel_metadata["cta"]["context"]["associated_elements"][0]["id"] == "MSTS-01"

    corsika_file = Path(tmp_test_directory) / "gamma.corsika.zst"
    corsika_file.write_bytes(b"")
    mocker.patch.object(
        event_metadata_reader,
        "get_corsika_run_and_event_headers",
        return_value=(
            np.array((42, 7.8010), dtype=[("run_number", "i4"), ("version", "f4")]),
            np.array((1, 14), dtype=[("event_number", "i4"), ("particle_id", "i4")]),
        ),
    )
    metadata_1.args_dict["input_meta"] = corsika_file
    corsika_metadata = metadata_1._read_input_metadata_from_file()[0]
    assert corsika_metadata["cta"]["product"]["filename"] == str(corsika_file)
    assert corsika_metadata["cta"]["product"]["format"] == "corsika"
    assert corsika_metadata["cta"]["activity"]["software"] == {
        "name": "corsika",
        "version": "7.8010",
    }
    assert '"run_number": 42' in corsika_metadata["cta"]["context"]["notes"][0]["text"]

    metadata_1.args_dict["input_meta"] = simple_test_file
    with pytest.raises(ValueError, match=r"^Unknown metadata file format:"):
        metadata_1._read_input_metadata_from_file()


def test_read_input_metadata_from_ecsv(args_dict_site, caplog):
    metadata_1 = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(FileNotFoundError):
            metadata_1._read_input_metadata_from_ecsv("file_not_there.ecsv")
    assert "Failed reading metadata for" in caplog.text


def test_merge_config_dicts(args_dict_site):
    d_low_priority = {
        "reference": {"version": "1.0.0"},
        "activity": {"name": "SetParameterFromExternal", "description": "Set data columns"},
        "datamodel": "model-A",
        "product": None,
    }

    d_high_priority = {
        "reference": {"version": "0.2.0"},
        "activity": {"name": None},
        "product": {"directory": "./"},
        "datamodel": "model-B",
    }
    d_high_priority_2 = copy.deepcopy(d_high_priority)
    # this should do only some thing with d_high_priority!
    _metadata = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    _metadata._merge_config_dicts(d_high_priority, d_low_priority, add_new_fields=True)

    d_merged = {
        "reference": {"version": "0.2.0"},
        "activity": {"name": "SetParameterFromExternal", "description": "Set data columns"},
        "product": {"directory": "./"},
        "datamodel": "model-B",
    }
    assert d_merged == d_high_priority

    d_merged_no_adding = {
        "reference": {"version": "0.2.0"},
        "activity": {"name": "SetParameterFromExternal"},
        "product": {"directory": "./"},
        "datamodel": "model-B",
    }
    _metadata._merge_config_dicts(d_high_priority_2, d_low_priority, add_new_fields=False)

    assert d_merged_no_adding == d_high_priority_2

    d_high_priority = None
    with pytest.raises(TypeError, match="Error merging dictionaries"):
        _metadata._merge_config_dicts(d_high_priority, d_low_priority, add_new_fields=None)


def test_fill_activity_meta(args_dict_site):
    file_writer_1 = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    file_writer_1._fill_activity_meta(file_writer_1.top_level_meta["cta"]["activity"])

    # this is set by args_dict_site in conf.py  (although this is a unit test)
    assert file_writer_1.top_level_meta["cta"]["activity"]["name"] == "integration_test"


def get_generic_input_meta():
    return {
        "contact": "my_name",
        "instrument": "my_instrument",
        "product": {
            "description": "my_product",
            "create_time": "2050-01-01",
        },
        "process": "process_description",
        "context": {
            "associated_elements": [
                {"site": "South", "class": "MST", "type": "FlashCam", "subtype": "D", "id:": None},
                {"site": "North", "class": "MST", "type": "NectarCam", "subtype": "7", "id:": None},
            ],
        },
    }


def get_example_input_schema_single_parameter():
    return {
        "version": "1.0.0",
        "name": "ref_lat",
        "description": "Latitude of site centre.",
        "short_description": "Latitude of site centre.",
        "data": [{"type": "double", "units": "deg", "allowed_range": {"min": -90.0, "max": 90.0}}],
    }


def test_fill_context_meta_associated_data_from_args(args_dict_site):
    collector = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    collector.input_metadata = None

    # associated_data present in args_dict: entries appended
    entry = {
        "id": "C25EE35E-8F9B-42C4-BFE4-96D9E1DB8256",
        "description": "CTAO subarray identifiers",
    }
    collector.args_dict = dict(args_dict_site)
    collector.args_dict["associated_data"] = [entry]
    context_dict = {"associated_data": []}
    collector._fill_context_meta(context_dict)
    assert context_dict["associated_data"] == [entry]

    # associated_data absent in args_dict: list stays empty
    collector.args_dict = dict(args_dict_site)
    context_dict = {"associated_data": []}
    collector._fill_context_meta(context_dict)
    assert context_dict["associated_data"] == []

    # associated_data key absent in context_dict: no error raised
    collector.args_dict["associated_data"] = [entry]
    context_dict = {}
    collector._fill_context_meta(context_dict)
    assert "associated_data" not in context_dict


def test_fill_context_meta_notes_from_value_table_without_meta(args_dict_site, tmp_test_directory):
    table_file = tmp_test_directory / "test_input.ecsv"
    table = Table({"x": [1.0, 2.0]})
    table.write(table_file, format="ascii.ecsv")

    collector = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    collector.input_metadata = None
    collector.args_dict = dict(args_dict_site)
    collector.args_dict["value"] = str(table_file)

    context_dict = {"notes": []}
    collector._fill_context_meta(context_dict)
    assert context_dict["notes"] == []


def test_fill_context_meta_notes_from_value_table_with_data_path(
    args_dict_site, tmp_test_directory
):
    table_file = tmp_test_directory / "test_input.ecsv"
    table = Table({"x": [1.0, 2.0]})
    table.meta["context_from_sim_telarray"] = "simtel context"
    table.write(table_file, format="ascii.ecsv")

    collector = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    collector.input_metadata = None
    collector.args_dict = dict(args_dict_site)
    collector.args_dict["value"] = "test_input.ecsv"
    collector.args_dict["data_path"] = str(tmp_test_directory)

    context_dict = {"notes": []}
    collector._fill_context_meta(context_dict)

    assert len(context_dict["notes"]) == 1
    assert context_dict["notes"][0]["title"] == "context_from_sim_telarray"
    assert context_dict["notes"][0]["text"] == "simtel context"


def test_write_metadata_to_yml(args_dict_site, tmp_test_directory, caplog):
    collector = metadata_collector.MetadataCollector(args_dict=args_dict_site)

    with pytest.raises(TypeError, match="No output file for metadata defined"):
        collector.write(yml_file=None)

    with caplog.at_level(logging.INFO):
        yml_file = collector.write(yml_file=tmp_test_directory.join("test_file.yml"))
    assert "Writing metadata to" in caplog.text
    assert Path(yml_file).exists()

    yml_file = collector.write(
        yml_file=tmp_test_directory.join("test_file.yml"), add_activity_name=True
    )
    assert Path(yml_file).exists()
    assert yml_file.name == "test_file.integration_test.meta.yml"

    with pytest.raises(FileNotFoundError, match=r"^Error writing metadata"):
        collector.write(yml_file="./this_directory_is_not_there/test_file.yml")

    with pytest.raises(TypeError, match="No output file for metadata defined"):
        collector.write(yml_file=None)


def test_fill_contact_meta_failed_system(args_dict_site, caplog, monkeypatch):

    def mock_getuser():
        raise KeyError("Failed to get username")

    monkeypatch.setattr(getpass, "getuser", mock_getuser)

    contact_dict = {}
    collector = metadata_collector.MetadataCollector(args_dict=args_dict_site)

    with caplog.at_level(logging.WARNING):
        collector._fill_contact_meta(contact_dict)

    assert "Failed to get user name" in caplog.text
    assert contact_dict["name"] == "UNKNOWN_USER"


def test_read_input_metadata_from_yml_or_json_no_file():
    collector = metadata_collector.MetadataCollector(args_dict={})
    with pytest.raises(FileNotFoundError, match=r"Failed reading metadata from missing_file\.yml"):
        collector._read_input_metadata_from_yml_or_json("missing_file.yml")
