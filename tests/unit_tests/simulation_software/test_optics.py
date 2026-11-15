"""Tests for the optics request of the simulation chain."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from simtools.simtel.shower_photons import derive_seed
from simtools.simulation_software import optics
from simtools.simulation_software.optics import OpticsRequest

_NOT_IN_RANGE = "must lie in"


@pytest.fixture
def request_one():
    return OpticsRequest(
        telescope_type="TOY-01",
        telescope_name="TOYN-02",
        telescope_id=3,
        model_version="9.9.9",
        site="North",
        dual_mirror=False,
        zenith_deg=20.0,
        azimuth_deg=180.0,
        instrument_seed=3,
        tracing_seed=4,
    )


def _array_model(types_by_name, seed_string="11,22"):
    models = {
        name: SimpleNamespace(
            name=type_name,
            get_parameter_value=lambda _name, mirror_class=mirror_class: mirror_class,
        )
        for name, (type_name, mirror_class) in types_by_name.items()
    }
    return SimpleNamespace(
        telescope_models=models,
        model_version="7.0.0",
        site="South",
        sim_telarray_seed=SimpleNamespace(seed_string=seed_string, instrument_seed=None),
    )


def test_seeds_of_seed_string():
    seeds = SimpleNamespace(seed_string="11,22", instrument_seed=None)
    assert optics.seeds_of(seeds) == (11, 22)


def test_seeds_of_seed_file():
    seeds = SimpleNamespace(
        seed_string="file-by-run:seeds.txt", instrument_seed="5", simulation_seed="6"
    )
    assert optics.seeds_of(seeds) == (5, 6)


def test_seeds_of_seed_file_without_instrument_seed():
    seeds = SimpleNamespace(seed_string="file-by-run:seeds.txt", instrument_seed=None)
    with pytest.raises(ValueError, match="Instrument seed is not set"):
        optics.seeds_of(seeds)


def test_request_is_frozen(request_one):
    with pytest.raises(AttributeError):
        request_one.threads = 2


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"telescope_transmission": "none"}, "Invalid telescope transmission"),
        ({"telescope_transmission": "angular"}, "stay unapplied"),
        ({"telescope_id": -1}, "telescope id"),
        ({"zenith_deg": 90.0}, _NOT_IN_RANGE),
        ({"zenith_deg": -1.0}, _NOT_IN_RANGE),
        ({"tracing_seed": -1}, "must not be negative"),
        ({"threads": -1}, "must not be negative"),
    ],
)
def test_request_validation(request_one, change, match):
    with pytest.raises(ValueError, match=match):
        replace(request_one, **change)


def test_optics_requests_one_per_telescope():
    array_model = _array_model(
        {"LSTS-01": ("LST", 1), "MSTS-01": ("MST", 2), "LSTS-02": ("LST", 1)}
    )
    parameters = SimpleNamespace(zenith_angle=20.0, azimuth_angle=180.0)
    ids = {"LSTS-01": 0, "MSTS-01": 1, "LSTS-02": 2}

    requests = optics.optics_requests(array_model, parameters, ids, stream_id_offset=1, threads=2)

    assert [request.telescope_name for request in requests] == ["LSTS-01", "MSTS-01", "LSTS-02"]
    assert [request.telescope_type for request in requests] == ["LST", "MST", "LST"]
    assert [request.telescope_id for request in requests] == [1, 2, 3]
    assert [request.dual_mirror for request in requests] == [False, True, False]
    first = requests[0]
    assert (first.model_version, first.site) == ("7.0.0", "South")
    assert (first.zenith_deg, first.azimuth_deg) == (20.0, 180.0)
    assert (first.tracing_seed, first.telescope_transmission, first.threads) == (22, "full", 2)


def test_instrument_seed_of_each_telescope_is_derived_from_its_iact_id():
    array_model = _array_model({"LSTS-01": ("LST", 1), "LSTS-02": ("LST", 1)})
    parameters = SimpleNamespace(zenith_angle=20.0, azimuth_angle=180.0)
    ids = {"LSTS-01": 0, "LSTS-02": 1}

    first, second = optics.optics_requests(array_model, parameters, ids, stream_id_offset=1)

    assert first.instrument_seed == derive_seed(11, 0)
    assert second.instrument_seed == derive_seed(11, 1)
    assert first.instrument_seed != second.instrument_seed
    assert first.model_file_name != second.model_file_name


def test_write_telescope_map(request_one, tmp_test_directory):
    path = optics.write_telescope_map([request_one], Path(tmp_test_directory) / "map.json")

    entries = json.loads(path.read_text(encoding="utf-8"))
    assert list(entries) == ["3"]
    assert entries["3"]["name"] == "TOYN-02"
    assert entries["3"]["model"] == "TOY-01"
    assert entries["3"]["phi_deg"] == pytest.approx(180.0)
