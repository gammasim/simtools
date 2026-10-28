"""Tests for CORSIKA metadata adaptation."""

import pytest

from simtools.sim_events import metadata_reader


def test_read_metadata_rejects_missing_headers(mocker):
    mocker.patch.object(
        metadata_reader,
        "get_corsika_run_and_event_headers",
        return_value=(None, None),
    )
    with pytest.raises(ValueError, match=r"^CORSIKA file has no complete run and event header:"):
        metadata_reader.read_metadata("gamma.corsika.zst")
