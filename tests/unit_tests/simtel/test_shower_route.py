#!/usr/bin/python3

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from simtools.corsika.photon_bunches import iter_photon_objects, read_photon_bunches
from simtools.simtel import shower_route
from simtools.simtel.photon_bunch_file import (
    BUNCH_DTYPE,
    PhotonBunchEvent,
    _block,
    _object,
    _write_telescope_block,
    write_photon_bunches,
)
from simtools.simtel.shower_route import (
    MultipipeWakeup,
    RouteInputs,
    core_reuses,
    derive_seed,
    load_telescope_map,
    main,
    parse_telescope_map,
    route_stream,
    select_routes,
    transmission_correction,
)

THETA_DEG = 20.0
PHI_DEG = 35.0
SEED = 4242
OBSERVATION_LEVEL_M = 2147.0
RADIUS_M = 2.0
INPUT_PLANE_HEIGHT_M = 30.0
INCOMING_OBSCURER_TOP_M = 27.1
ZEM_CM = 216350.0
FAKE_POSITION_M = np.array([0.1, 0.2, 16.0])
FAKE_DIRECTION = np.array([0.0, 0.0, -1.0])
VALID_ENTRY = {
    "model": "LSTS-01",
    "model_version": "7.0.0",
    "site": "South",
    "dual_mirror": False,
    "radius_m": 12.5,
    "input_plane_height_m": 30.0,
    "theta_deg": THETA_DEG,
    "phi_deg": PHI_DEG,
}


class _FakeOpticalModel:
    """Optical model returning one camera photon per input bunch, recording its calls."""

    def __init__(self):
        self.calls = []

    def max_incoming_obscurer_z_m(self):
        return INCOMING_OBSCURER_TOP_M

    def trace_bunches(self, bunches, **kwargs):
        self.calls.append((len(bunches["x"]), kwargs))
        n_bunches = len(bunches["x"])
        return {
            "position_m": np.tile(FAKE_POSITION_M, (n_bunches, 1)),
            "direction": np.tile(FAKE_DIRECTION, (n_bunches, 1)),
            "time_ns": bunches["time_ns"].astype(np.float64),
            "weight": np.ones(n_bunches),
            "wavelength_nm": bunches["wavelength_nm"].astype(np.float64),
            "bunch_index": np.arange(n_bunches, dtype=np.int64),
            "sub_photon": np.zeros(n_bunches, dtype=np.int64),
        }


class _FakeTracer:
    """Tracer returning one fake optical model and recording the optical-model requests."""

    def __init__(self):
        self.model = _FakeOpticalModel()
        self.requests = []

    def optical_model(self, theta_deg, seed):
        self.requests.append((theta_deg, seed))
        return self.model


def _atmospheric_profile():
    return Table(
        {
            "altitude": [0.0, 2.0, 10.0] * u.km,
            "density": [1.2e-3, 1.0e-3, 4.0e-4] * u.g / u.cm**3,
            "thickness": [1030.0, 800.0, 270.0] * u.g / u.cm**2,
            "refractive_index": [2.8e-4, 2.3e-4, 1.0e-4],
        }
    )


def _bunches(n_bunches):
    bunches = np.zeros(n_bunches, dtype=BUNCH_DTYPE)
    bunches["x"] = np.arange(n_bunches) * 10.0
    bunches["y"] = -np.arange(n_bunches) * 10.0
    bunches["cx"], bunches["cy"] = 0.01, -0.02
    bunches["time"] = 55.0
    bunches["zem"] = ZEM_CM
    bunches["photons"] = 0.5
    bunches["wavelength"] = 385.0
    return bunches


def _stream(blocks, event_numbers=None):
    """IACT stream with one event per (telescope id, bunches) block, in order."""
    event_numbers = event_numbers or range(1, len(blocks) + 1)
    events = [
        PhotonBunchEvent(number=number, blocks=((telescope_id, bunches),))
        for number, (telescope_id, bunches) in zip(event_numbers, blocks, strict=True)
    ]
    buffer = io.BytesIO()
    write_photon_bunches(
        buffer,
        events,
        observation_level_m=OBSERVATION_LEVEL_M,
        telescope_radius_m=RADIUS_M,
        atmospheric_profile=_atmospheric_profile(),
    )
    return io.BytesIO(buffer.getvalue())


def _route_kwargs(telescope_ids=(0,), input_plane_height_m=INPUT_PLANE_HEIGHT_M):
    return {
        "theta_deg_by_telescope": dict.fromkeys(telescope_ids, THETA_DEG),
        "phi_deg": PHI_DEG,
        "input_plane_height_m_by_telescope": dict.fromkeys(telescope_ids, input_plane_height_m),
        "seed": SEED,
    }


def _fake_inputs_factory(routes, *, simulation_models_path=None, backend=None, context=None):
    """Route inputs with fake tracers (replaces the telescope models in the CLI test)."""
    del simulation_models_path, backend, context
    return RouteInputs(
        tracers={telescope_id: _FakeTracer() for telescope_id in routes},
        observation_level_m=OBSERVATION_LEVEL_M,
        atmospheric_profile=_atmospheric_profile(),
    )


def _read_back(data):
    return list(read_photon_bunches(io.BytesIO(data)))


def test_route_stream_writes_one_event_per_telescope_block():
    stream = _stream(
        [(0, _bunches(3)), (1, _bunches(2)), (0, _bunches(0)), (1, _bunches(4))],
        event_numbers=(7, 7, 8, 9),
    )
    tracers = {0: _FakeTracer(), 1: _FakeTracer()}
    out = io.BytesIO()

    count = route_stream(stream, out, tracers, **_route_kwargs(telescope_ids=(0, 1)))

    # Two blocks with event number 7 are the telescope blocks of one shower: one event header.
    assert count == 3
    events = _read_back(out.getvalue())
    assert [len(event["x"]) for event in events] == [3, 2, 0, 4]
    # Output events keep the telescope id and the event number of their input blocks.
    assert [event["telescope_id"] for event in events] == [0, 1, 0, 1]
    assert [event["event_number"] for event in events] == [7, 7, 8, 9]
    # One optical model per telescope id, at the telescope zenith angle and its own seed.
    assert tracers[0].requests == [(THETA_DEG, derive_seed(SEED, 0))]
    assert tracers[1].requests == [(THETA_DEG, derive_seed(SEED, 1))]
    # Each telescope is traced with its own tracer; the empty block is not traced.
    assert [calls for calls, _ in tracers[0].model.calls] == [3]
    assert [calls for calls, _ in tracers[1].model.calls] == [2, 4]


def test_route_stream_keeps_telescope_blocks_of_every_event():
    # Pixel lists need one photon block per telescope in every event; no event is dropped.
    events = [
        PhotonBunchEvent(number=number, blocks=((0, _bunches(2 + number)), (1, _bunches(3))))
        for number in (1, 2, 3)
    ]
    buffer = io.BytesIO()
    write_photon_bunches(
        buffer,
        events,
        observation_level_m=OBSERVATION_LEVEL_M,
        telescope_radius_m=RADIUS_M,
        atmospheric_profile=_atmospheric_profile(),
    )
    out = io.BytesIO()

    count = route_stream(
        io.BytesIO(buffer.getvalue()),
        out,
        {0: _FakeTracer(), 1: _FakeTracer()},
        **_route_kwargs(telescope_ids=(0, 1)),
    )

    assert count == 3
    blocks = [
        (event["event_number"], event["telescope_id"], len(event["x"]))
        for event in _read_back(out.getvalue())
    ]
    assert blocks == [
        (1, 0, 3),
        (1, 1, 3),
        (2, 0, 4),
        (2, 1, 3),
        (3, 0, 5),
        (3, 1, 3),
    ]


def test_route_stream_passes_tracing_arguments():
    stream = _stream([(0, _bunches(2))])
    tracer = _FakeTracer()

    route_stream(stream, io.BytesIO(), {0: tracer}, **_route_kwargs(), threads=3)

    _, kwargs = tracer.model.calls[0]
    # Tracing seed of the block: event number 1, array id 0, telescope id 0.
    assert kwargs["seed"] == derive_seed(SEED, 1, 0, 0)
    assert kwargs["threads"] == 3
    assert kwargs["roulette"] is True
    assert kwargs["normalise_transmission"] is False
    assert kwargs["plane_z"] == pytest.approx(100.0 * INPUT_PLANE_HEIGHT_M)


def test_derive_seed_is_reproducible_and_depends_on_keys():
    seed = derive_seed(SEED, 3, 0, 7)

    assert seed == derive_seed(SEED, 3, 0, 7)
    assert 0 <= seed < 2**32
    assert derive_seed(SEED, 7) != derive_seed(SEED, 8)
    assert derive_seed(SEED, 3, 0, 7) != derive_seed(SEED, 3, 1, 7)
    assert derive_seed(SEED, 3, 0, 7) != derive_seed(SEED, 4, 0, 7)
    assert derive_seed(SEED, 7) != derive_seed(SEED + 1, 7)


def test_camera_events_tracing_seed_differs_by_telescope_and_array(monkeypatch):
    # Blocks of one event: telescope 1, telescope 2, telescope 1 of another array, and a
    # repeat of the first block (same event, array and telescope: same seed).
    blocks = list(
        read_photon_bunches(
            _stream(
                [(1, _bunches(2)), (2, _bunches(2)), (1, _bunches(2)), (1, _bunches(2))],
                event_numbers=(5, 5, 5, 5),
            )
        )
    )
    # The writer numbers the blocks of an event by position; set the array ids explicitly.
    for block, array_id in zip(blocks, (0, 0, 1, 0), strict=True):
        block["array_id"] = array_id
    objects = [
        (SimpleNamespace(header=SimpleNamespace(id=block["array_id"])), b"", [block])
        for block in blocks
    ]
    monkeypatch.setattr(shower_route, "iter_photon_objects", lambda _, **_kwargs: iter(objects))
    tracers = {1: _FakeTracer(), 2: _FakeTracer()}
    frame = shower_route.shower_bunch_frame(
        THETA_DEG, PHI_DEG, input_plane_height_m=INPUT_PLANE_HEIGHT_M
    )
    telescopes = {
        telescope_id: shower_route._TracedTelescope(
            model=tracer.model, frame=frame, dual_mirror=False
        )
        for telescope_id, tracer in tracers.items()
    }

    events = list(
        shower_route._camera_events(
            None, telescopes, skipped_telescope_ids=frozenset(), seed=SEED, threads=0
        )
    )

    assert len(events) == 4
    seeds = [
        tracers[1].model.calls[0][1]["seed"],
        tracers[2].model.calls[0][1]["seed"],
        tracers[1].model.calls[1][1]["seed"],
        tracers[1].model.calls[2][1]["seed"],
    ]
    assert seeds[0] == derive_seed(SEED, 5, 0, 1)
    assert seeds[1] == derive_seed(SEED, 5, 0, 2)
    assert seeds[2] == derive_seed(SEED, 5, 1, 1)
    # Same inputs, same seed.
    assert seeds[3] == seeds[0]
    assert len(set(seeds[:3])) == 3


def test_route_stream_unknown_telescope_id_raises():
    stream = _stream([(0, _bunches(2)), (2, _bunches(2))])

    with pytest.raises(ValueError, match="Telescope id 2 of event 2 is not in the telescope map"):
        route_stream(stream, io.BytesIO(), {0: _FakeTracer()}, **_route_kwargs())


def test_route_stream_empty_event_is_written_and_not_traced():
    stream = _stream([(0, _bunches(0))])
    tracer = _FakeTracer()
    out = io.BytesIO()

    count = route_stream(stream, out, {0: tracer}, **_route_kwargs())

    assert count == 1
    events = _read_back(out.getvalue())
    assert [len(event["x"]) for event in events] == [0]
    assert tracer.model.calls == []


def test_route_stream_input_plane_below_minimum_writes_nothing():
    stream = _stream([(0, _bunches(2))])
    out = io.BytesIO()

    with pytest.raises(ValueError, match="below the minimum"):
        route_stream(stream, out, {0: _FakeTracer()}, **_route_kwargs(input_plane_height_m=20.0))
    assert out.getvalue() == b""


def test_route_stream_skipped_telescope_writes_event_without_blocks():
    stream = _stream(
        [(0, _bunches(2)), (1, _bunches(3)), (0, _bunches(1))], event_numbers=(1, 2, 3)
    )
    tracer = _FakeTracer()
    out = io.BytesIO()

    count = route_stream(
        stream, out, {0: tracer}, **_route_kwargs(telescope_ids=(0,)), skipped_telescope_ids={1}
    )

    # The event of the skipped telescope is written without telescope blocks, so the reader
    # (which lists telescope blocks) shows two telescope blocks.
    assert count == 3
    events = _read_back(out.getvalue())
    assert [event["event_number"] for event in events] == [1, 3]
    assert [len(event["x"]) for event in events] == [2, 1]
    # The block of the skipped telescope is not traced.
    assert [calls for calls, _ in tracer.model.calls] == [2, 1]


def test_route_stream_telescope_neither_traced_nor_skipped_raises():
    with pytest.raises(ValueError, match="Telescope id 1 of event 1 is not in the telescope map"):
        route_stream(
            _stream([(1, _bunches(1))]),
            io.BytesIO(),
            {0: _FakeTracer()},
            **_route_kwargs(telescope_ids=(0,)),
            skipped_telescope_ids={2},
        )


def test_select_routes_all_and_subset():
    routes = parse_telescope_map({"1": VALID_ENTRY, "2": VALID_ENTRY})

    assert set(select_routes(routes)) == {1, 2}
    assert set(select_routes(routes, [2])) == {2}
    with pytest.raises(ValueError, match="Telescope ids \\[5\\] are not in the telescope map"):
        select_routes(routes, [5])


def test_route_stream_missing_telescope_parameter_raises():
    with pytest.raises(ValueError, match="No zenith angle for telescope id 1"):
        route_stream(
            _stream([(0, _bunches(1))]),
            io.BytesIO(),
            {0: _FakeTracer(), 1: _FakeTracer()},
            **_route_kwargs(telescope_ids=(0,)),
        )


def test_route_stream_real_corsika_file_through_pipe():
    iact_file = os.environ.get("SIMTOOLS_TEST_IACT_FILE")
    if not iact_file:
        pytest.skip("Set SIMTOOLS_TEST_IACT_FILE to a CORSIKA IACT file with photon bunches")
    expected = [len(event["x"]) for event in read_photon_bunches(iact_file)]

    # Identity tracer: one camera photon per input bunch; the output keeps every block.
    with subprocess.Popen(["cat", iact_file], stdout=subprocess.PIPE) as cat:
        out = io.BytesIO()
        count = route_stream(cat.stdout, out, {0: _FakeTracer()}, **_route_kwargs())

    # The count is the number of showers (event headers); each shower has several telescope blocks.
    showers = sum(1 for obj, _, _ in iter_photon_objects(iact_file) if obj.header.type == 1202)
    assert count == showers
    events = _read_back(out.getvalue())
    assert [len(event["x"]) for event in events] == expected


def test_parse_telescope_map_valid():
    routes = parse_telescope_map({"0": VALID_ENTRY, "3": {**VALID_ENTRY, "dual_mirror": True}})

    assert set(routes) == {0, 3}
    assert routes[0].name == "LSTS-01"
    assert routes[0].model_version == "7.0.0"
    assert routes[0].site == "South"
    assert routes[0].dual_mirror is False
    assert routes[0].radius_m == pytest.approx(12.5)
    assert routes[0].input_plane_height_m == pytest.approx(30.0)
    assert routes[3].dual_mirror is True


def test_load_telescope_map_reads_json(tmp_test_directory):
    path = Path(tmp_test_directory) / "telescope_map.json"
    path.write_text(json.dumps({"0": VALID_ENTRY}), encoding="utf-8")

    routes = load_telescope_map(path)

    assert routes[0].name == "LSTS-01"


@pytest.mark.parametrize("key", list(VALID_ENTRY))
def test_parse_telescope_map_missing_key_raises(key):
    entry = {name: value for name, value in VALID_ENTRY.items() if name != key}

    with pytest.raises(ValueError, match=f"lacks {key}"):
        parse_telescope_map({"0": entry})


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({}, "non-empty object"),
        ([VALID_ENTRY], "non-empty object"),
        ({"LST": VALID_ENTRY}, "not an integer"),
        ({"0": "LSTS-01"}, "is not an object"),
        ({"0": {**VALID_ENTRY, "site": "East"}}, "site must be one of"),
        ({"0": {**VALID_ENTRY, "dual_mirror": "no"}}, "dual_mirror must be true or false"),
        ({"0": {**VALID_ENTRY, "model": ""}}, "model must be a non-empty string"),
        ({"0": {**VALID_ENTRY, "model_version": 7}}, "model_version must be a non-empty string"),
        ({"0": {**VALID_ENTRY, "radius_m": -1.0}}, "radius_m must be a positive number"),
        ({"0": {**VALID_ENTRY, "radius_m": True}}, "radius_m must be a positive number"),
        ({"0": {**VALID_ENTRY, "input_plane_height_m": "30"}}, "must be a positive number"),
        ({"1": VALID_ENTRY, "01": VALID_ENTRY}, "occurs twice"),
    ],
)
def test_parse_telescope_map_invalid_raises(mapping, message):
    with pytest.raises(ValueError, match=message):
        parse_telescope_map(mapping)


def test_load_telescope_map_invalid_json_raises(tmp_test_directory):
    path = Path(tmp_test_directory) / "telescope_map.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        load_telescope_map(path)


def test_cli_main_streams_stdin_to_stdout(tmp_test_directory):
    map_path = Path(tmp_test_directory) / "telescope_map.json"
    map_path.write_text(json.dumps({"0": VALID_ENTRY}), encoding="utf-8")
    stream = _stream([(0, _bunches(3)), (0, _bunches(0)), (0, _bunches(2))]).getvalue()
    code = (
        "import sys; from test_shower_route import _fake_inputs_factory; "
        "from simtools.simtel.shower_route import main; "
        "sys.exit(main(sys.argv[1:], inputs_factory=_fake_inputs_factory))"
    )

    # The subprocess must import the simtools sources of this tree, not an installed copy.
    src_dir = str(Path(shower_route.__file__).parents[2])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([src_dir, os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run(
        [sys.executable, "-c", code, "--telescope-map", str(map_path)],
        input=stream,
        capture_output=True,
        cwd=Path(__file__).parent,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert len(result.stdout) > 0
    assert [len(event["x"]) for event in _read_back(result.stdout)] == [3, 0, 2]


def test_main_returns_zero_in_process(tmp_test_directory, monkeypatch):
    map_path = Path(tmp_test_directory) / "telescope_map.json"
    map_path.write_text(json.dumps({"0": VALID_ENTRY}), encoding="utf-8")
    stdout = io.BytesIO()
    stdin = SimpleNamespace(buffer=_stream([(0, _bunches(1))]))
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=stdout))

    status = main(
        ["--telescope-map", str(map_path)],
        inputs_factory=_fake_inputs_factory,
    )

    assert status == 0
    assert len(_read_back(stdout.getvalue())) == 1


def test_core_reuses_from_card():
    assert core_reuses(["NSHOW 5", "CSCAT 10 30000.0 0.0", "EXIT"]) == 10
    assert core_reuses(["NSHOW 5"]) == 1
    assert core_reuses(["CSCAT 0 30000.0 0.0"]) == 1


_MULTIPIPE_ENV = {
    "MULTIPIPE_SEQUENTIAL": "1",
    "MULTIPIPE_PID": "4711",
    "MULTIPIPE_SIGNAL": "18",
    "MULTIPIPE_SIGNAL_INIT": "34",
}


def _toplevel(object_type):
    return SimpleNamespace(header=SimpleNamespace(type=object_type))


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {k: v for k, v in _MULTIPIPE_ENV.items() if k != "MULTIPIPE_SEQUENTIAL"},
        {**_MULTIPIPE_ENV, "MULTIPIPE_PID": "1"},
        {**_MULTIPIPE_ENV, "MULTIPIPE_PID": "not-a-pid"},
        {**_MULTIPIPE_ENV, "MULTIPIPE_SIGNAL": "0"},
        {**_MULTIPIPE_ENV, "MULTIPIPE_SIGNAL_INIT": "-1"},
    ],
)
def test_multipipe_wakeup_needs_valid_sequential_environment(environ):
    assert MultipipeWakeup.from_environment(environ, send=None) is None


def test_multipipe_wakeup_from_environment_reads_signals():
    sent = []
    wakeup = MultipipeWakeup.from_environment(_MULTIPIPE_ENV, send=lambda *a: sent.append(a))

    assert wakeup is not None
    wakeup.start()
    assert sent == [(4711, 34)]


def test_multipipe_wakeup_sends_one_signal_per_object_after_event_header():
    sent = []
    wakeup = MultipipeWakeup.from_environment(_MULTIPIPE_ENV, send=lambda *a: sent.append(a))
    # Run header, input card, telescope positions (no wakeup), atmospheric profile (not counted),
    # then the first event header and every following object (one wakeup each).
    for object_type in (1200, 1212, 1216, 1201, 1202, 1203, 1204, 1204, 1209, 1210):
        wakeup.on_object(_toplevel(object_type))

    assert sent == [(4711, 18)] * 6


def test_multipipe_wakeup_atmospheric_profile_after_event_header_is_signalled():
    sent = []
    wakeup = MultipipeWakeup.from_environment(_MULTIPIPE_ENV, send=lambda *a: sent.append(a))
    for object_type in (1200, 1201, 1202, 1216):
        wakeup.on_object(_toplevel(object_type))

    assert len(sent) == 2


def test_multipipe_wakeup_send_failure_is_logged_not_raised(caplog):
    def _failing_send(_pid, _signal):
        raise ProcessLookupError("gone")

    wakeup = MultipipeWakeup.from_environment(_MULTIPIPE_ENV, send=_failing_send)
    wakeup.start()
    wakeup.on_object(_toplevel(1202))

    assert "failed" in caplog.text


def test_route_stream_signals_each_processed_object():
    stream = _stream([(0, _bunches(3)), (0, _bunches(2))], event_numbers=(1, 2))
    sent = []
    types = []
    wakeup = MultipipeWakeup.from_environment(_MULTIPIPE_ENV, send=lambda *a: sent.append(a))
    wakeup.start()

    def on_object(obj):
        types.append(obj.header.type)
        wakeup.on_object(obj)

    route_stream(
        stream,
        io.BytesIO(),
        {0: _FakeTracer()},
        **_route_kwargs(telescope_ids=(0,)),
        on_object=on_object,
    )

    # Input objects of the CORSIKA-like stream: run header, telescope positions, atmospheric
    # profile, then per event an event header, telescope offsets and a telescope object, and
    # the run end. Only the first three objects before the first event header get no wakeup.
    assert types[:4] == [1200, 1201, 1216, 1202]
    assert sent[0] == (4711, 34)
    assert sent[1:] == [(4711, 18)] * (len(types) - 3)


def test_route_preserves_shower_objects_and_reuses():
    import struct

    source = io.BytesIO()
    source.write(_object(1200, 42, _block(b"RUNH", {2: 42})))
    source.write(_object(1212, 0, b"card"))
    source.write(_object(1201, 0, struct.pack("<i", 0)))
    # Word 46 holds the CORSIKA version, which selects the event header layout of corsikaio.
    source.write(_object(1202, 17, _block(b"EVTH", {2: 17, 4: 1250, 11: 0.4, 12: 1.2, 46: 6.999})))
    source.write(_object(1203, 0, struct.pack("<ifffff", 2, 0, 120, -40, 80, 60)))
    source.write(_object(1211, 17, b"long"))
    for reuse in (0, 1):
        subobjects = io.BytesIO()
        for telescope_id in (0, 1):
            block = io.BytesIO()
            _write_telescope_block(block, telescope_id, _bunches(2), reuse=reuse)
            subobjects.write(block.getvalue()[16:])
        source.write(_object(1204, reuse, subobjects.getvalue(), only_subobjects=True))
    source.write(_object(1209, 17, _block(b"EVTE", {2: 17})))
    source.write(_object(1210, 42, _block(b"RUNE", {2: 42})))
    source.write(_object(1234, 7, b"other"))
    original = source.getvalue()
    output = io.BytesIO()
    callbacks = []

    count = route_stream(
        io.BytesIO(original),
        output,
        {0: _FakeTracer()},
        **_route_kwargs(),
        skipped_telescope_ids={1},
        on_object=lambda obj: callbacks.append(obj.header.type),
    )

    before = list(iter_photon_objects(io.BytesIO(original)))
    after = list(iter_photon_objects(io.BytesIO(output.getvalue())))
    assert count == 1
    assert callbacks == [obj.header.type for obj, _, _ in before]
    assert [obj.header.type for obj, _, _ in after] == callbacks
    for (old_obj, old_raw, _), (new_obj, new_raw, blocks) in zip(before, after, strict=True):
        if old_obj.header.type == 1204:
            assert new_obj.header.id == old_obj.header.id
            assert len(blocks) == 1
            assert blocks[0]["array_id"] == old_obj.header.id
            assert blocks[0]["telescope_id"] == 0
            assert new_raw != old_raw
        else:
            assert new_raw == old_raw


def test_route_zeroes_only_traced_telescope_heights():
    import struct

    from simtools.simtel.shower_route import _zero_traced_telescope_heights

    payload = struct.pack("<i12f", 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    raw = _object(1201, 0, payload)
    result = _zero_traced_telescope_heights(raw, {0: object(), 2: object()})
    expected = struct.pack("<i12f", 3, 1, 2, 3, 4, 5, 6, 0, 8, 0, 10, 11, 12)
    assert result == _object(1201, 0, expected)


def test_route_atmospheric_correction_and_disable(tmp_test_directory):
    from simtools.simtel.shower_transmission import SimtelTransmission, transmission_ratio

    path = Path(tmp_test_directory) / "transmission.dat"
    path.write_text("# H2= 2.147, H1= 3 10 30\n385 0.1 0.3 0.5\n", encoding="utf-8")
    table = SimtelTransmission(path, OBSERVATION_LEVEL_M)
    bunches = _bunches(1)
    bunches["cx"], bunches["cy"] = np.sin(np.deg2rad(60)), 0
    bunches["zem"] = 900000
    original = _stream([(0, bunches)]).getvalue()
    objects = list(iter_photon_objects(io.BytesIO(original)))
    import struct

    data = b"".join(
        _object(1201, 0, struct.pack("<i4f", 1, 0, 0, 3400, 200))
        if obj.header.type == 1201
        else raw
        for obj, raw, _ in objects
    )
    results = []
    for transmission in (None, table):
        output = io.BytesIO()
        route_stream(
            io.BytesIO(data),
            output,
            {0: _FakeTracer()},
            **_route_kwargs(),
            correction=transmission_correction(transmission, {0: 28.0}),
        )
        results.append(_read_back(output.getvalue())[0]["size"][0])
    ground_airmass = 1 / np.sqrt(1 - float(bunches["cx"][0]) ** 2)
    expected = transmission_ratio(table, 385, 9000, ground_airmass, 1, 34, 0, 28, 2147)
    assert results[0] == pytest.approx(1.0)
    assert results[1] == pytest.approx(expected, rel=1e-6)


def test_route_drops_zero_bypass_transmission(tmp_test_directory, caplog):
    from simtools.simtel.shower_transmission import SimtelTransmission

    path = Path(tmp_test_directory) / "transmission.dat"
    path.write_text("# H2= 2.147, H1= 3 10 30\n385 0.1 0.3 0.5\n", encoding="utf-8")
    table = SimtelTransmission(path, OBSERVATION_LEVEL_M)
    bunches = _bunches(1)
    bunches["zem"] = 100000
    output = io.BytesIO()
    import logging

    with caplog.at_level(logging.INFO, logger=shower_route.__name__):
        route_stream(
            _stream([(0, bunches)]),
            output,
            {0: _FakeTracer()},
            **_route_kwargs(),
            correction=transmission_correction(table, {0: 28.0}),
        )
    assert len(_read_back(output.getvalue())[0]["size"]) == 0
    assert "Dropped 1 bunches with zero bypass transmission" in caplog.text


def test_atmospheric_correction_cli_default_and_override():
    assert not shower_route.parse_args(["--telescope-map", "map.json"]).no_atmospheric_correction
    assert shower_route.parse_args(
        ["--telescope-map", "map.json", "--no-atmospheric-correction"]
    ).no_atmospheric_correction
