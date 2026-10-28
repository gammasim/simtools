"""Tests for sim_telarray metadata adaptation."""

import pytest

from simtools.simtel import metadata_reader


def test_read_metadata_rejects_invalid_registry_value(mocker):
    mocker.patch.object(
        metadata_reader,
        "read_sim_telarray_metadata",
        return_value=({"azimuth_angle": "not-a-number"}, {}),
    )
    with pytest.raises(ValueError, match=r"could not convert string to float"):
        metadata_reader.read_metadata("gamma.simtel.zst")


def test_read_metadata_without_metadata(mocker):
    mocker.patch.object(metadata_reader, "read_sim_telarray_metadata", return_value=(None, None))
    fields = metadata_reader.read_metadata("empty.simtel")

    assert fields["software_name"] == "sim_telarray"
    assert fields["site"] is None
    assert fields["associated_elements"] == []
