"""Tests for merging event-aligned telescope streams."""

import numpy as np
import pytest

from simtools.simulation_software import merge_streams

stream = pytest.importorskip("obdeect.stream")


def _write_input(path, telescope_id, event_numbers=(1, 2), weight=1.0):
    with stream.Writer(str(path), "test") as writer:
        writer.write("run", {"run_number": 42, "n_telescopes": 1, "custom": "preserved"})
        for event_number in event_numbers:
            writer.write("event", {"event_number": event_number, "label": "shared"})
            writer.write("telescope", {"telescope_id": telescope_id, "frame": 3})
            for start in (0, 1):
                columns = {
                    name: np.array([weight + start], dtype=np.float32)
                    for name in ("x", "y", "cx", "cy", "t", "w")
                }
                writer.write_photons(telescope_id, "camera", start, columns)
            writer.write("telescope_end", {"telescope_id": telescope_id, "photons": weight * 2})
            writer.write("event_end", {"event_number": event_number})
        writer.write("run_end", {})


def test_merge_streams_preserves_photons_and_metadata(tmp_test_directory):
    first = tmp_test_directory / "first.pst"
    second = tmp_test_directory / "second.pst"
    output = tmp_test_directory / "merged.pst"
    _write_input(first, 1, weight=1.0)
    _write_input(second, 2, weight=2.0)

    assert merge_streams.merge_streams([first, second], output) == 2

    records = list(stream.Reader(str(output)))
    run = next(record for record in records if record["type"] == "run")
    assert run["metadata"] == {"run_number": 42, "n_telescopes": 1, "custom": "preserved"}
    assert [
        record["metadata"]["telescope_id"] for record in records if record["type"] == "telescope"
    ] == [1, 2, 1, 2]
    chunks = [record for record in records if record["type"] == "photons"]
    assert [record["first_index"] for record in chunks] == [0, 1] * 4
    assert [float(record["columns"]["w"][0]) for record in chunks] == [1, 2, 2, 3] * 2
    ends = [record["metadata"] for record in records if record["type"] == "telescope_end"]
    assert [record["photons"] for record in ends] == [2, 4, 2, 4]
    assert [
        record["metadata"]["event_number"] for record in records if record["type"] == "event_end"
    ] == [1, 2]


def test_merge_streams_rejects_different_event_sequences(tmp_test_directory):
    first = tmp_test_directory / "first.pst"
    second = tmp_test_directory / "second.pst"
    _write_input(first, 1)
    _write_input(second, 2, event_numbers=(1, 3))

    with pytest.raises(ValueError, match="event sequence"):
        merge_streams.merge_streams([first, second], tmp_test_directory / "merged.pst")


def test_merge_reuses_and_preserves_scoped_status_and_blobs(tmp_test_directory):
    files = [tmp_test_directory / f"tel-{tel}.pst" for tel in (2, 5)]
    for path, telescope in zip(files, (2, 5)):
        with stream.Writer(str(path), "test") as writer:
            writer.write("run", {"run_number": 7, "telescope_x_m": [0.0, 10.0]})
            writer.write_blob(1200, 7, b"RUNH")
            writer.write(
                "efficiency_status",
                {
                    "factors": "mirror_reflectivity",
                    "mirror_reflectivity.state": "applied",
                    "mirror_reflectivity.source": f"model-{telescope}",
                },
            )
            writer.write("event", {"event_number": 12, "core_x_m": [0.0, 1.0]})
            writer.write_blob(1202, 12, b"EVTH")
            for index in (0, 1):
                writer.write(
                    "telescope", {"telescope_id": telescope, "frame": 3, "array_index": index}
                )
                columns = {
                    name: np.array([index], dtype=np.float32)
                    for name in ("x", "y", "cx", "cy", "t", "w")
                }
                writer.write_photons(telescope, "camera", index, columns)
                writer.write("telescope_end", {"telescope_id": telescope})
            writer.write_blob(1209, 12, b"EVTE")
            writer.write("event_end", {})
            writer.write_blob(1210, 7, b"RUNE")
            writer.write("run_end", {})
    output = tmp_test_directory / "merged.pst"
    merge_streams.merge_streams(files, output)
    records = list(stream.Reader(str(output)))
    assert [r["data"] for r in records if r["type"] == "blob"] == [
        b"RUNH",
        b"EVTH",
        b"EVTE",
        b"RUNE",
    ]
    types = [r["type"] for r in records]
    assert types.index("event") < next(i for i, r in enumerate(records) if r.get("data") == b"EVTH")
    assert next(i for i, r in enumerate(records) if r.get("data") == b"EVTH") < types.index(
        "telescope"
    )
    assert (
        max(i for i, kind in enumerate(types) if kind == "telescope_end")
        < next(i for i, r in enumerate(records) if r.get("data") == b"EVTE")
        < types.index("event_end")
    )
    assert [
        (r["metadata"]["array_index"], r["metadata"]["telescope_id"])
        for r in records
        if r["type"] == "telescope"
    ] == [(0, 2), (0, 5), (1, 2), (1, 5)]
    assert [
        r["metadata"]["mirror_reflectivity.source"]
        for r in records
        if r["type"] == "efficiency_status"
    ] == ["model-2", "model-5", "model-2", "model-5"]
    assert [r["first_index"] for r in records if r["type"] == "photons"] == [0, 0, 1, 1]


def test_rejects_duplicate_telescope_ids(tmp_test_directory):
    first = tmp_test_directory / "first.pst"
    second = tmp_test_directory / "second.pst"
    _write_input(first, 1)
    _write_input(second, 1)
    with pytest.raises(ValueError, match="telescope IDs are repeated"):
        merge_streams.merge_streams([first, second], tmp_test_directory / "merged.pst")


def test_rejects_different_run_metadata(tmp_test_directory):
    first = tmp_test_directory / "first.pst"
    second = tmp_test_directory / "second.pst"
    _write_input(first, 1)
    with stream.Writer(str(second), "test") as writer:
        writer.write("run", {"run_number": 100})
        writer.write("run_end", {})
    with pytest.raises(ValueError, match="different run metadata"):
        merge_streams.merge_streams([first, second], tmp_test_directory / "merged.pst")


def test_empty_runs_preserve_run_trailer(tmp_test_directory):
    paths = [tmp_test_directory / "first.pst", tmp_test_directory / "second.pst"]
    for path in paths:
        with stream.Writer(str(path), "test") as writer:
            writer.write("run", {"run_number": 7})
            writer.write_blob(1210, 7, b"RUNE")
            writer.write("run_end", {"events": 0})
    output = tmp_test_directory / "merged.pst"
    assert merge_streams.merge_streams(paths, output) == 0
    records = list(stream.Reader(str(output)))
    assert [r["type"] for r in records] == ["stream", "run", "blob", "run_end"]
    assert records[-1]["metadata"] == {"events": 0}


def test_merge_requires_two_input_streams(tmp_test_directory):
    with pytest.raises(ValueError, match="at least two"):
        merge_streams.merge_streams([], tmp_test_directory / "merged.pst")


def test_merge_rejects_different_event_blobs(tmp_test_directory):
    paths = [tmp_test_directory / "first.pst", tmp_test_directory / "second.pst"]
    for path, telescope in zip(paths, (1, 2)):
        with stream.Writer(str(path), "test") as writer:
            writer.write("run", {"run_number": 7})
            writer.write("event", {"event_number": 1})
            writer.write_blob(1202, 1, bytes([telescope]))
            writer.write("event_end", {})
            writer.write("run_end", {})
    with pytest.raises(ValueError, match="different event blobs"):
        merge_streams.merge_streams(paths, tmp_test_directory / "merged.pst")


def test_command_line_merges_streams(tmp_test_directory):
    paths = [tmp_test_directory / "first.pst", tmp_test_directory / "second.pst"]
    _write_input(paths[0], 1)
    _write_input(paths[1], 2)
    output = tmp_test_directory / "merged.pst"
    assert merge_streams.main(["--output", str(output), *(str(path) for path in paths)]) == 0
    assert sum(r["type"] == "event" for r in stream.Reader(str(output))) == 2


def test_rejects_different_event_trailers(tmp_test_directory):
    paths = [tmp_test_directory / "first.pst", tmp_test_directory / "second.pst"]
    for path, telescope in zip(paths, (1, 2)):
        with stream.Writer(str(path), "test") as writer:
            writer.write("run", {"run_number": 7})
            writer.write("event", {"event_number": 1})
            writer.write("event_end", {"number": telescope})
            writer.write("run_end", {})
    with pytest.raises(ValueError, match="different event_end metadata"):
        merge_streams.merge_streams(paths, tmp_test_directory / "merged.pst")


def test_rejects_mismatched_telescope_end(tmp_test_directory):
    paths = [tmp_test_directory / "first.pst", tmp_test_directory / "second.pst"]
    for path, telescope in zip(paths, (1, 2)):
        with stream.Writer(str(path), "test") as writer:
            writer.write("run", {"run_number": 7})
            writer.write("event", {"event_number": 1})
            writer.write("telescope", {"telescope_id": telescope, "frame": 3})
            writer.write("telescope_end", {"telescope_id": telescope + 1})
            writer.write("event_end", {})
            writer.write("run_end", {})
    with pytest.raises(ValueError, match="telescope_end ID does not match"):
        merge_streams.merge_streams(paths, tmp_test_directory / "merged.pst")


def test_rejects_missing_run_end(tmp_test_directory):
    paths = [tmp_test_directory / "first.pst", tmp_test_directory / "second.pst"]
    for path in paths:
        with stream.Writer(str(path), "test") as writer:
            writer.write("run", {"run_number": 7})
    with pytest.raises(ValueError, match="missing run_end"):
        merge_streams.merge_streams(paths, tmp_test_directory / "merged.pst")


def test_rejects_mismatched_run_blobs(tmp_test_directory):
    paths = [tmp_test_directory / "first.pst", tmp_test_directory / "second.pst"]
    for index, path in enumerate(paths):
        with stream.Writer(str(path), "test") as writer:
            writer.write("run", {"run_number": 7})
            writer.write_blob(1200, 7, bytes([index]))
            writer.write("run_end", {})
    with pytest.raises(ValueError, match="different run metadata or blobs"):
        merge_streams.merge_streams(paths, tmp_test_directory / "merged.pst")
