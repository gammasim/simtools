#!/usr/bin/python3

from unittest import mock

import pytest

import simtools.simtel.simtel_io_metadata as simtel_io_metadata
from simtools.simtel.simtel_io_metadata import (
    _decode_dictionary,
    _guess_telescope_name_for_legacy_files,
    get_sim_telarray_telescope_id,
    get_sim_telarray_telescope_id_to_telescope_name_mapping,
    read_sim_telarray_metadata,
)


def test_decode_success():
    test_meta = {b"key1": b"value1", b"key2": b"value2"}
    result = _decode_dictionary(test_meta)
    assert result == {"key1": "value1", "key2": "value2"}
    result = _decode_dictionary(test_meta, encoding="ascii")
    assert result == {"key1": "value1", "key2": "value2"}


def test_decode_with_unicode_error(caplog):
    # Create metadata with invalid unicode bytes
    test_meta = {b"key1": b"value1", b"key2": b"\xff\xfe invalid utf8"}

    result = _decode_dictionary(test_meta, encoding="utf-8")

    assert "key1" in result
    assert "key2" in result
    assert result["key1"] == "value1"
    assert result["key2"] == " invalid utf8"
    assert "Unable to decode metadata with encoding utf-8" in caplog.text


def test_read_sim_telarray_metadata(sim_telarray_file_gamma):
    global_meta, telescope_meta = read_sim_telarray_metadata(sim_telarray_file_gamma)
    assert global_meta is not None
    assert len(telescope_meta) > 0
    assert isinstance(telescope_meta, dict)
    assert all(isinstance(k, int) for k in telescope_meta.keys())
    assert all(isinstance(v, dict) for v in telescope_meta.values())

    for key in global_meta.keys():
        assert key[0] != "*"
        assert key.strip() == key
        assert key.lower() == key

    assert (float)(global_meta["latitude"]) > 0.0
    assert global_meta["array_config_name"] == "test_layout"


@mock.patch.object(simtel_io_metadata, "_decode_dictionary", return_value=None, autospec=True)
def test_read_sim_telarray_metadata_attribute_error(mock_decode, sim_telarray_file_gamma):
    simtel_io_metadata.read_sim_telarray_metadata.cache_clear()
    with pytest.raises(AttributeError, match=r"^Error reading metadata from file"):
        read_sim_telarray_metadata(sim_telarray_file_gamma)


def test_get_sim_telarray_telescope_id(sim_telarray_file_gamma):
    assert get_sim_telarray_telescope_id("LSTN-01", sim_telarray_file_gamma) == 1
    assert get_sim_telarray_telescope_id("MSTN-01", sim_telarray_file_gamma) == 5
    assert get_sim_telarray_telescope_id("MSTS-01", sim_telarray_file_gamma) is None


def test_get_sim_telarray_telescope_id_to_telescope_name_mapping(sim_telarray_file_gamma):
    tel_mapping = get_sim_telarray_telescope_id_to_telescope_name_mapping(sim_telarray_file_gamma)

    assert isinstance(tel_mapping, dict)
    assert len(tel_mapping) > 0
    assert all(isinstance(k, int) for k in tel_mapping.keys())
    assert all(isinstance(v, str) for v in tel_mapping.values())

    assert tel_mapping[1] == "LSTN-01"
    assert tel_mapping[5] == "MSTN-01"


def test_get_telescope_list_from_input_card_parses_telescopes(monkeypatch):
    simtel_io_metadata._get_telescope_list_from_input_card.cache_clear()

    # Simulate InputCard object with parse() returning a string matching the regex
    class FakeInputCard:
        def parse(self):
            return b"""
                TELESCOPE    -70.91E2     -52.35E2 45.00E2  12.50E2  # (ID=1)  LSTN   01   2B5\n
                TELESCOPE   -153.29E2     168.86E2 28.70E2  9.15E2  # (ID=6)   MSTN   02   4B1\n
                TELESCOPE   -153.29E2     168.86E2 28.70E2  9.15E2  # (ID=6)   LSTS   02   4B1\n
                TELESCOPE   -153.29E2     168.86E2 28.70E2  9.15E2  # (ID=6)   MSTS   02   4B1\n
                TELESCOPE   -153.29E2     168.86E2 28.70E2  9.15E2  # (ID=6)   SSTS   02   4B1\n
                TELESCOPE   -153.29E2     168.86E2 28.70E2  9.15E2  # (ID=6)   SCTS   02   4B1\n
                """

    class FakeEventIOFile:
        def __init__(self, *args, **kwargs):  # test init
            pass

        def __enter__(self):
            return [FakeInputCard()]

        def __exit__(self, exc_type, exc_val, exc_tb):  # test exit
            pass

    monkeypatch.setattr(simtel_io_metadata, "EventIOFile", FakeEventIOFile)
    monkeypatch.setattr(simtel_io_metadata, "InputCard", FakeInputCard)
    result = simtel_io_metadata._get_telescope_list_from_input_card("dummy1.simtel")
    assert isinstance(result, list)
    assert "LSTN-01" in result
    assert "MSTN-02" in result
    assert "LSTS-02" in result
    assert "MSTS-02" in result
    assert "SSTS-02" in result
    assert "SCTS-02" in result
    assert len(result) == 6


def test_get_telescope_list_from_input_card_no_input_card(monkeypatch):
    simtel_io_metadata._get_telescope_list_from_input_card.cache_clear()

    class FakeInputCard:
        def parse(self):
            # No telescope lines matching the regex
            return b""

    class FakeEventIOFile:
        def __init__(self, *args, **kwargs):  # test init
            pass

        def __enter__(self):
            # No InputCard objects
            return []

        def __exit__(self, exc_type, exc_val, exc_tb):  # test exit
            pass

    monkeypatch.setattr(simtel_io_metadata, "EventIOFile", FakeEventIOFile)
    monkeypatch.setattr(simtel_io_metadata, "InputCard", FakeInputCard)
    result = simtel_io_metadata._get_telescope_list_from_input_card("dummy2.simtel")
    assert result == []


def test_get_telescope_list_from_input_card_input_card_no_match(monkeypatch):
    simtel_io_metadata._get_telescope_list_from_input_card.cache_clear()

    class FakeInputCard:
        def parse(self):
            # No telescope lines matching the regex
            return b"TELESCOPE -70.91E2 -52.35E2 45.00E2 12.50E2 # (ID=1) ACT 01 2B5\n"

    class FakeEventIOFile:
        def __init__(self, *args, **kwargs):  # test init
            pass

        def __enter__(self):
            return [FakeInputCard()]

        def __exit__(self, exc_type, exc_val, exc_tb):  # test exit
            pass

    monkeypatch.setattr(simtel_io_metadata, "EventIOFile", FakeEventIOFile)
    result = simtel_io_metadata._get_telescope_list_from_input_card("dummy.simtel")
    assert result == []


def test_guess_telescope_name_for_legacy_files(monkeypatch):
    # Patch _get_telescope_list_from_input_card to return a known list
    monkeypatch.setattr(
        "simtools.simtel.simtel_io_metadata._get_telescope_list_from_input_card",
        lambda file: ["LSTN-01", "MSTN-02", "SSTC-03"],
    )

    # Should return the correct validated name for index 1
    result = _guess_telescope_name_for_legacy_files(1, "dummy5.simtel")
    assert result == "MSTN-02"

    # Should return None for out-of-range index
    result_none = _guess_telescope_name_for_legacy_files(10, "dummy5.simtel")
    assert result_none is None


def test_get_sim_telarray_telescope_id_to_telescope_name_mapping_value_error(monkeypatch, mocker):
    # Patch validate_array_element_name to always raise ValueError
    mocker.patch(
        "simtools.utils.names.validate_array_element_name",
        side_effect=ValueError("invalid name"),
    )
    # Patch _guess_telescope_name_for_legacy_files to return a fallback name
    monkeypatch.setattr(
        "simtools.simtel.simtel_io_metadata._guess_telescope_name_for_legacy_files",
        lambda idx, file: f"FAKE-{idx}",
    )
    # Patch read_sim_telarray_metadata to return dummy telescope_meta
    monkeypatch.setattr(
        "simtools.simtel.simtel_io_metadata.read_sim_telarray_metadata",
        lambda file: ({}, {1: {"optics_config_name": "bad"}, 2: {"optics_config_name": "bad2"}}),
    )
    mapping = get_sim_telarray_telescope_id_to_telescope_name_mapping("dummy4.simtel")
    assert mapping == {1: "FAKE-0", 2: "FAKE-1"}
