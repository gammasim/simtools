#!/usr/bin/python3


from simtools.simtel.simtel_io_metadata import (
    _decode_dictionary,
    get_sim_telarray_telescope_id,
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
    assert "Failed to decode metadata with encoding utf-8" in caplog.text


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


def test_get_sim_telarray_telescope_id(sim_telarray_file_gamma):
    assert get_sim_telarray_telescope_id("LSTN-01", sim_telarray_file_gamma) == 1
    assert get_sim_telarray_telescope_id("MSTN-01", sim_telarray_file_gamma) == 5
    assert get_sim_telarray_telescope_id("MSTS-01", sim_telarray_file_gamma) is None
