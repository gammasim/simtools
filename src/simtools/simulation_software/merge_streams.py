"""Merge event-aligned telescope photon streams for an array simulation."""

import argparse
import itertools
import sys

import numpy as np

_SPLIT_MARKERS = {1213, 1214}


def _same(left, right):
    """Compare stream metadata, including float arrays."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple, np.ndarray)) or isinstance(right, (list, tuple, np.ndarray)):
        return bool(np.array_equal(left, right, equal_nan=True))
    return left == right


def _write(writer, record):
    """Copy a stream record, dropping CORSIKA split markers."""
    kind = record["type"]
    if kind == "photons":
        writer.write_photons(
            record["telescope_id"], record["frame"], record["first_index"], record["columns"]
        )
    elif kind == "blob":
        if record["eventio_type"] not in _SPLIT_MARKERS:
            writer.write_blob(record["eventio_type"], record["eventio_id"], record["data"])
    else:
        writer.write(kind, record["metadata"])


def _preamble(reader, path):
    """Read a stream header, run header, and records before the first event."""
    header = next(reader, None)
    run = next(reader, None)
    if not header or header["type"] != "stream" or not run or run["type"] != "run":
        raise ValueError(f"{path}: expected stream and run headers")
    records = []
    for record in reader:
        if record["type"] in {"event", "run_end"}:
            return run, records, itertools.chain((record,), reader)
        if record["type"] not in {"blob", "efficiency_status"}:
            raise ValueError(f"{path}: unexpected run record {record['type']!r}")
        records.append(record)
    raise ValueError(f"{path}: missing run_end")


def _next_event(reader, path):
    """Read one event or the run end, including preceding run-level blobs."""
    prefix = []
    for record in reader:
        kind = record["type"]
        if kind == "blob":
            prefix.append(record)
        elif kind in {"event", "run_end"}:
            break
        else:
            raise ValueError(f"{path}: unexpected record {kind!r} between events")
    else:
        raise ValueError(f"{path}: missing run_end")
    if kind == "run_end":
        return prefix, record, [], None
    return _read_event(reader, path, prefix, record)


def _read_event(reader, path, prefix, record):
    """Collect the telescope blocks of one event."""
    blocks = []
    event_prefix = []
    suffix = []
    for child in reader:
        child_kind = child["type"]
        if child_kind == "event_end":
            return prefix, record, blocks, (event_prefix, suffix, child)
        if child_kind == "telescope":
            if suffix:
                raise ValueError(f"{path}: non-marker blob between telescope blocks")
            blocks.append(_read_telescope(reader, path, child))
        elif child_kind == "blob":
            _event_blob(child, bool(blocks), event_prefix, suffix)
        else:
            raise ValueError(f"{path}: unexpected event record {child_kind!r}")
    raise ValueError(f"{path}: missing event_end")


def _read_telescope(reader, path, header):
    """Collect one complete telescope block."""
    block = [header]
    for record in reader:
        kind = record["type"]
        if kind == "telescope_end":
            if record["metadata"].get("telescope_id") != header["metadata"].get("telescope_id"):
                raise ValueError(f"{path}: telescope_end ID does not match telescope")
            return [*block, record]
        if kind not in {"photons", "efficiency_status"}:
            raise ValueError(f"{path}: unexpected record {kind!r} inside telescope")
        block.append(record)
    raise ValueError(f"{path}: missing telescope_end")


def _event_blob(record, seen_block, prefix, suffix):
    """Place an event blob around its telescope records."""
    if record["eventio_type"] in _SPLIT_MARKERS:
        return
    (suffix if seen_block else prefix).append(record)


def _blobs(records):
    """Identify raw EventIO blobs for comparing repeated input streams."""
    return [(r["eventio_type"], r["eventio_id"], r["data"]) for r in records if r["type"] == "blob"]


def _check_event(parts):
    """Verify matching event headers, trailers, and shared EventIO blobs."""
    prefix, header, _, end = parts[0]
    for other_prefix, other_header, _, other_end in parts[1:]:
        if header["type"] != other_header["type"] or not _same(
            header["metadata"], other_header["metadata"]
        ):
            raise ValueError("input streams have different event sequences or run metadata")
        if _blobs(prefix) != _blobs(other_prefix) or (end is None) != (other_end is None):
            raise ValueError("input streams have different event sequences or blobs")
        if end:
            _check_event_end(end, other_end)


def _check_event_end(end, other_end):
    """Verify event-level blobs and end metadata."""
    if _blobs(end[0]) != _blobs(other_end[0]) or _blobs(end[1]) != _blobs(other_end[1]):
        raise ValueError("input streams have different event blobs")
    if not _same(end[2]["metadata"], other_end[2]["metadata"]):
        raise ValueError("input streams have different event_end metadata")


def _write_blocks(writer, parts, statuses):
    """Write the telescope blocks grouped by array reuse, retaining their own status."""
    blocks = []
    seen = set()
    for index, (_, _, telescope_blocks, _) in enumerate(parts):
        ids = {block[0]["metadata"]["telescope_id"] for block in telescope_blocks}
        if seen.intersection(ids):
            raise ValueError("telescope IDs are repeated across input streams")
        seen.update(ids)
        blocks.extend((block, statuses[index]) for block in telescope_blocks)
    blocks.sort(key=lambda item: item[0][0]["metadata"].get("array_index", 0))
    for block, inherited in blocks:
        _write(writer, block[0])
        for status in inherited:
            _write(writer, status)
        for record in block[1:]:
            _write(writer, record)


def merge_streams(inputs, output):
    """Merge telescope streams from the same run, preserving event and telescope records."""
    import obdeect.stream  # pylint: disable=import-outside-toplevel

    if len(inputs) < 2:
        raise ValueError("at least two input streams are required")
    streams = [_preamble(iter(obdeect.stream.Reader(str(path))), path) for path in inputs]
    run, preamble, _ = streams[0]
    for other_run, other_preamble, _ in streams[1:]:
        if not _same(run["metadata"], other_run["metadata"]) or _blobs(preamble) != _blobs(
            other_preamble
        ):
            raise ValueError("input streams have different run metadata or blobs")
    statuses = [
        [r for r in records if r["type"] == "efficiency_status"] for _, records, _ in streams
    ]
    with obdeect.stream.Writer(str(output), "simtools-merge-streams") as writer:
        _write(writer, run)
        for record in preamble:
            if record["type"] == "blob":
                _write(writer, record)
        return _write_events(writer, inputs, streams, statuses)


def _write_events(writer, inputs, streams, statuses):
    """Write matched events and the run trailer."""
    event_count = 0
    while True:
        parts = [_next_event(reader, path) for path, (_, _, reader) in zip(inputs, streams)]
        _check_event(parts)
        prefix, header, _, end = parts[0]
        for record in prefix:
            _write(writer, record)
        _write(writer, header)
        if end is None:
            if any(next(reader, None) is not None for _, _, reader in streams):
                raise ValueError("input contains records after run_end")
            return event_count
        for record in end[0]:
            _write(writer, record)
        _write_blocks(writer, parts, statuses)
        for record in end[1]:
            _write(writer, record)
        _write(writer, end[2])
        event_count += 1


def main(argv=None):
    """Run the photon-stream merger."""
    parser = argparse.ArgumentParser(description="Merge event-aligned telescope streams.")
    parser.add_argument("--output", required=True, help="Output photon-stream file.")
    parser.add_argument("inputs", nargs="+", help="Input telescope photon streams.")
    args = parser.parse_args(argv)
    merge_streams(args.inputs, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
