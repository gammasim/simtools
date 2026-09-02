#!/usr/bin/python3

from types import SimpleNamespace
from unittest import mock

import pytest

import simtools.simtel.simtel_io_metadata as simtel_io_metadata


class _FakeHistoryMeta:
    def __init__(self, metadata, telescope_id):
        self.header = SimpleNamespace(id=telescope_id)
        self._metadata = metadata

    def parse(self):
        return self._metadata


def test_decode_success():
    test_meta = {b"key1": b"value1", b"key2": b"value2"}
    result = simtel_io_metadata._decode_dictionary(test_meta)
    assert result == {"key1": "value1", "key2": "value2"}
    result = simtel_io_metadata._decode_dictionary(test_meta, encoding="ascii")
    assert result == {"key1": "value1", "key2": "value2"}


def test_decode_with_unicode_error(caplog):
    # Create metadata with invalid unicode bytes
    test_meta = {b"key1": b"value1", b"key2": b"\xff\xfe invalid utf8"}

    result = simtel_io_metadata._decode_dictionary(test_meta, encoding="utf-8")

    assert "key1" in result
    assert "key2" in result
    assert result["key1"] == "value1"
    assert result["key2"] == " invalid utf8"
    assert "Unable to decode metadata with encoding utf-8" in caplog.text


def test_read_sim_telarray_metadata(mocker):
    simtel_io_metadata.read_sim_telarray_metadata.cache_clear()
    mocker.patch.object(simtel_io_metadata, "HistoryMeta", _FakeHistoryMeta)
    eventio_file = mocker.patch.object(simtel_io_metadata, "EventIOFile")
    eventio_file.return_value.__enter__.return_value = [
        _FakeHistoryMeta(
            {
                b"*Latitude": b" 28.0 ",
                b"Array_Config_Name": b" CTAO-North-Alpha ",
                b"simtools_simtel_version": b"v2025-11-30-rc",
            },
            -1,
        ),
        _FakeHistoryMeta(
            {
                b"Optics_Config_Variant": b"LSTN-01 ",
                b"Camera_Config_Variant": b"LSTN-01 ",
            },
            1,
        ),
        object(),
    ]

    global_meta, telescope_meta = simtel_io_metadata.read_sim_telarray_metadata(
        "synthetic.simtel.zst"
    )
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
    assert global_meta["array_config_name"] == "CTAO-North-Alpha"
    assert global_meta["simtools_simtel_tag"] == "v2025-11-30-rc"
    assert "simtools_simtel_version" not in global_meta


def test_normalize_sim_telarray_metadata_maps_legacy_names():
    metadata = {
        "simtools_corsika_config_version": "v0.1.0",
        "simtools_corsika_opt_patch_version": "v1.1.0",
        "simtools_corsika_version": "78010",
        "simtools_hessio_version": "v2025-12-01-rc",
        "simtools_simtel_version": "v2025-11-30-rc",
        "simtools_stdtools_version": "v2025-06-16-rc",
        "custom_metadata": "preserved",
    }

    normalized = simtel_io_metadata.normalize_sim_telarray_metadata(metadata)

    assert normalized == {
        "simtools_corsika_config_tag": "v0.1.0",
        "simtools_corsika_opt_patch_tag": "v1.1.0",
        "simtools_corsika_build_id": "78010",
        "simtools_hessio_tag": "v2025-12-01-rc",
        "simtools_simtel_tag": "v2025-11-30-rc",
        "simtools_stdtools_tag": "v2025-06-16-rc",
        "custom_metadata": "preserved",
    }


def test_normalize_sim_telarray_metadata_resolves_corsika_source_tag():
    normalized = simtel_io_metadata.normalize_sim_telarray_metadata(
        {"simtools_corsika_version": "78010"},
        dependency_catalog={"corsika": [{"tag": "v7.8010"}]},
    )

    assert normalized == {
        "simtools_corsika_build_id": "78010",
        "simtools_corsika_source_tag": "v7.8010",
    }


def test_normalize_sim_telarray_metadata_rejects_conflicting_aliases():
    with pytest.raises(ValueError, match="Conflicting sim_telarray metadata values"):
        simtel_io_metadata.normalize_sim_telarray_metadata(
            {
                "simtools_simtel_version": "v2025-11-30-rc",
                "simtools_simtel_tag": "v2025-12-01-rc",
            }
        )


@mock.patch.object(simtel_io_metadata, "_decode_dictionary", return_value=None, autospec=True)
def test_read_sim_telarray_metadata_attribute_error(mock_decode, mocker):
    simtel_io_metadata.read_sim_telarray_metadata.cache_clear()
    mocker.patch.object(simtel_io_metadata, "HistoryMeta", _FakeHistoryMeta)
    eventio_file = mocker.patch.object(simtel_io_metadata, "EventIOFile")
    eventio_file.return_value.__enter__.return_value = [_FakeHistoryMeta({}, -1)]
    with pytest.raises(AttributeError, match=r"^Error reading metadata from file"):
        simtel_io_metadata.read_sim_telarray_metadata("synthetic.simtel.zst")


def test_get_sim_telarray_telescope_id(mocker):
    mocker.patch.object(
        simtel_io_metadata,
        "read_sim_telarray_metadata",
        return_value=(
            {},
            {
                1: {
                    "optics_config_variant": "LSTN-01",
                    "camera_config_variant": "LSTN-01",
                },
                5: {
                    "optics_config_variant": "MSTN-01",
                    "camera_config_variant": "MSTN-01",
                },
            },
        ),
    )
    assert simtel_io_metadata.get_sim_telarray_telescope_id("LSTN-01", "synthetic.simtel.zst") == 1
    assert simtel_io_metadata.get_sim_telarray_telescope_id("MSTN-01", "synthetic.simtel.zst") == 5
    assert (
        simtel_io_metadata.get_sim_telarray_telescope_id("MSTS-01", "synthetic.simtel.zst") is None
    )


def test_get_sim_telarray_telescope_id_to_telescope_name_mapping(mocker):
    mocker.patch.object(
        simtel_io_metadata,
        "read_sim_telarray_metadata",
        return_value=(
            {},
            {
                1: {"optics_config_variant": "LSTN-01"},
                5: {"optics_config_variant": "MSTN-01"},
            },
        ),
    )
    tel_mapping = simtel_io_metadata.get_sim_telarray_telescope_id_to_telescope_name_mapping(
        "synthetic.simtel.zst"
    )
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
                TELESCOPE   -153.29E2     168.86E2 28.70E2  9.15E2  # (ID=6)   MST2   01   4B1\n
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
    assert "MST2-01" in result
    assert len(result) == 7


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
    result = simtel_io_metadata._guess_telescope_name_for_legacy_files(1, "dummy5.simtel")
    assert result == "MSTN-02"

    # Should return None for out-of-range index
    result_none = simtel_io_metadata._guess_telescope_name_for_legacy_files(10, "dummy5.simtel")
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
    mapping = simtel_io_metadata.get_sim_telarray_telescope_id_to_telescope_name_mapping(
        "dummy4.simtel"
    )
    assert mapping == {1: "FAKE-0", 2: "FAKE-1"}


@pytest.mark.parametrize(
    ("msts", "expected"),
    [
        (
            ["LSTS-01", "MSTS-01", "MSTS-133", "SSTS-01"],
            ["LSTS-01", "MSTS-01", "MSTS-133", "SSTS-01"],
        ),
        (
            ["LSTS-01", "MSTS-01", "MSTS-133", "MST2-05", "SSTS-01"],
            ["LSTS-01", "MSTS-01", "MSTS-133", "MSTS-134", "SSTS-01"],
        ),
        ([], []),
        (["LSTS-01", "SSTS-01"], ["LSTS-01", "SSTS-01"]),
    ],
)
def test_legacy_merge_msts(msts, expected):
    assert simtel_io_metadata._legacy_merge_msts(msts) == expected
