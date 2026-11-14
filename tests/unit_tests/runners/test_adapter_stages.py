"""Tests for the adapter stages of the obdeect chain."""

import importlib.util
from types import SimpleNamespace

import astropy.units as u
import pytest

from simtools.runners import stages
from simtools.runners.adapter_stages import CorsikaToStreamStage, StreamToSimtelStage
from simtools.runners.chain_spec import (
    ATMOSPHERIC_EXTINCTION,
    MIRROR_REFLECTIVITY,
    TELESCOPE_TRANSMISSION,
    ChainSpec,
)
from simtools.runners.stages import STDIN, StageInputs, StreamTo
from simtools.simulation_chain.products import (
    CAMERA_PHOTONS_IACT,
    CHERENKOV_PHOTONS,
    PHOTONS_CAMERA,
    PHOTONS_GROUND,
)


class _Telescope:
    """Telescope model parameters used by the camera map."""

    def __init__(self, mirror_class, camera_rotate_deg=0.0):
        self._mirror_class = mirror_class
        self._camera_rotate_deg = camera_rotate_deg

    def get_parameter_value(self, name):
        assert name == "mirror_class"
        return self._mirror_class

    def get_parameter_value_with_unit(self, name):
        assert name == "camera_rotate"
        return self._camera_rotate_deg * u.deg


def _array_model(*telescopes):
    """Array model with telescopes named LSTN-01, MSTN-01, ... (in this order)."""
    names = ("LSTN-01", "MSTN-01", "MSTN-02")
    return SimpleNamespace(telescope_models=dict(zip(names, telescopes)))


def _inputs(config, array_models=None, chain_options=None, showers=True):
    """Stage inputs of the CORSIKA configuration fixture."""
    return StageInputs(
        array_models=array_models or (config.array_model,),
        simulation_parameters=(config.simulation_parameters,),
        shower_configurations=(config,) if showers else (),
        args={},
        context=config.context,
        chain_options=chain_options or {},
    )


@pytest.fixture
def corsika_to_stream(corsika_config_mock_array_model):
    (stage,) = CorsikaToStreamStage.instances(_inputs(corsika_config_mock_array_model))
    return stage


@pytest.fixture
def stream_to_simtel(corsika_config_mock_array_model):
    array_model = _array_model(_Telescope(1), _Telescope(2, 90.0))
    inputs = _inputs(
        corsika_config_mock_array_model,
        array_models=(array_model,),
        chain_options={
            "stream_to_simtel": {
                "allow_pending": "telescope_transmission_constant",
                "scaled_factors": ["qe"],
            }
        },
    )
    (stage,) = StreamToSimtelStage.instances(inputs)
    return stage


def _fake_stage(name, consumes, produces, factors=()):
    """Stage class with a name, products and the efficiency factors it applies."""

    class FakeStage:
        streams = True

        @classmethod
        def applied_factors(cls, options):
            return frozenset(factors)

    FakeStage.name = name
    FakeStage.consumes = consumes
    FakeStage.produces = produces
    return FakeStage


def test_corsika_to_stream_protocol():
    assert CorsikaToStreamStage.name == "corsika_to_stream"
    assert CorsikaToStreamStage.consumes == CHERENKOV_PHOTONS
    assert CorsikaToStreamStage.produces == PHOTONS_GROUND
    assert CorsikaToStreamStage.streams is True


def test_stream_to_simtel_protocol():
    assert StreamToSimtelStage.name == "stream_to_simtel"
    assert StreamToSimtelStage.consumes == PHOTONS_CAMERA
    assert StreamToSimtelStage.produces == CAMERA_PHOTONS_IACT
    assert StreamToSimtelStage.streams is True


def test_instances_require_shower_configuration(corsika_config_mock_array_model):
    inputs = _inputs(corsika_config_mock_array_model, showers=False)

    with pytest.raises(ValueError, match="needs a CORSIKA configuration"):
        CorsikaToStreamStage.instances(inputs)
    with pytest.raises(ValueError, match="needs a CORSIKA configuration"):
        StreamToSimtelStage.instances(inputs)


def test_stream_to_simtel_rejects_several_array_models(corsika_config_mock_array_model):
    inputs = _inputs(
        corsika_config_mock_array_model,
        array_models=(_array_model(_Telescope(1)), _array_model(_Telescope(2))),
    )

    with pytest.raises(ValueError, match="one array model per run; got 2"):
        StreamToSimtelStage.instances(inputs)


def test_corsika_to_stream_pipes_stdin_into_downstream_command(corsika_to_stream):
    lines = corsika_to_stream.prepare(
        run_number=1, input_file=STDIN, output=StreamTo(("sim_telarray -",))
    )

    assert len(lines) == 1
    assert lines[0].endswith("stream/bin/obdeect-stream-from-corsika - | sim_telarray -")


def test_corsika_to_stream_one_line_per_downstream_command(corsika_to_stream):
    lines = corsika_to_stream.prepare(
        run_number=1, input_file=STDIN, output=StreamTo(("first -", "second -"))
    )

    assert len(lines) == 2
    assert lines[0].endswith("obdeect-stream-from-corsika - | first -")
    assert lines[1].endswith("obdeect-stream-from-corsika - | second -")


def test_corsika_to_stream_writes_stream_file_without_downstream(corsika_to_stream):
    lines = corsika_to_stream.prepare(run_number=1, input_file="run.iact")

    assert len(lines) == 1
    assert lines[0].endswith(
        f"obdeect-stream-from-corsika run.iact > {corsika_to_stream.output_file(1)}"
    )


def test_corsika_to_stream_requires_input(corsika_to_stream):
    with pytest.raises(ValueError, match="reads the CORSIKA output"):
        corsika_to_stream.prepare(run_number=1, output=StreamTo(("sim_telarray -",)))


def test_corsika_to_stream_reports_missing_program(corsika_to_stream, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ValueError, match="install the package 'obdeect'"):
        corsika_to_stream.prepare(run_number=1, input_file=STDIN, output=StreamTo(("x -",)))


def test_corsika_to_stream_reports_missing_program_in_installed_package(
    corsika_to_stream, monkeypatch, tmp_test_directory
):
    package_location = SimpleNamespace(submodule_search_locations=[str(tmp_test_directory)])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: package_location)

    with pytest.raises(ValueError, match="install the package 'obdeect'"):
        corsika_to_stream.prepare(run_number=1, input_file=STDIN, output=StreamTo(("x -",)))


def test_corsika_to_stream_environment_and_resources(corsika_to_stream):
    assert corsika_to_stream.environment() == {}
    assert corsika_to_stream.get_resources(runtime=1.5)["runtime"] == pytest.approx(1.5)


def test_stream_to_simtel_command_with_options(stream_to_simtel):
    camera_map = stream_to_simtel.camera_map_file(1)

    lines = stream_to_simtel.prepare(
        run_number=1, input_file=STDIN, output=StreamTo(("sim_telarray -",))
    )

    assert len(lines) == 1
    assert (
        f"obdeect-stream-to-simtel --camera-map {camera_map} "
        "--allow-pending telescope_transmission_constant --scaled-factor qe - | sim_telarray -"
    ) in lines[0]


def test_stream_to_simtel_writes_camera_map_of_all_telescopes(stream_to_simtel):
    stream_to_simtel.prepare(run_number=1, input_file=STDIN, output=StreamTo(("x -",)))

    assert stream_to_simtel.camera_map_file(1).read_text(encoding="utf-8") == (
        "0 1 0.0\n1 2 90.0\n"
    )


def test_stream_to_simtel_writes_stream_file_without_downstream(stream_to_simtel):
    lines = stream_to_simtel.prepare(run_number=1, input_file=STDIN)

    assert len(lines) == 1
    assert lines[0].endswith(f"--scaled-factor qe - > {stream_to_simtel.output_file(1)}")


def test_stream_to_simtel_reads_file_input(stream_to_simtel):
    lines = stream_to_simtel.prepare(
        run_number=1, input_file="photons.pst", output=StreamTo(("x -",))
    )

    assert lines[0].endswith("--scaled-factor qe photons.pst | x -")


def test_stream_to_simtel_single_allow_pending_as_list(corsika_config_mock_array_model):
    inputs = _inputs(
        corsika_config_mock_array_model,
        array_models=(_array_model(_Telescope(1)),),
        chain_options={"stream_to_simtel": {"allow_pending": ["a", "b"]}},
    )
    (stage,) = StreamToSimtelStage.instances(inputs)

    lines = stage.prepare(run_number=1, input_file=STDIN, output=StreamTo(("x -",)))

    assert "--allow-pending a --allow-pending b -" in lines[0]


def test_stream_to_simtel_requires_input(stream_to_simtel):
    with pytest.raises(ValueError, match="reads a photon stream"):
        stream_to_simtel.prepare(run_number=1, output=StreamTo(("sim_telarray -",)))


def test_stream_to_simtel_environment_and_resources(stream_to_simtel):
    assert stream_to_simtel.environment() == {}
    assert stream_to_simtel.get_resources(runtime=2.0)["runtime"] == pytest.approx(2.0)


@pytest.fixture
def fake_atmosphere_and_optics(monkeypatch):
    """Fake atmosphere and optics stages (the programs of other worktrees) of the obdeect chain."""
    monkeypatch.setitem(
        stages.STAGES,
        "fake_atmosphere",
        lambda: _fake_stage(
            "fake_atmosphere", PHOTONS_GROUND, PHOTONS_GROUND, (ATMOSPHERIC_EXTINCTION,)
        ),
    )
    monkeypatch.setitem(
        stages.STAGES,
        "fake_optics",
        lambda: _fake_stage(
            "fake_optics",
            PHOTONS_GROUND,
            PHOTONS_CAMERA,
            (MIRROR_REFLECTIVITY, TELESCOPE_TRANSMISSION),
        ),
    )


_OBDEECT_CHAIN = (
    "corsika_to_stream",
    "fake_atmosphere",
    "fake_optics",
    "stream_to_simtel",
    "sim_telarray",
)


def test_obdeect_chain_with_bypass_mode_3_is_valid(fake_atmosphere_and_optics):
    spec = ChainSpec(_OBDEECT_CHAIN, {"sim_telarray": {"bypass_optics": 3}})

    chain = spec.chain()

    assert chain.stages == _OBDEECT_CHAIN


@pytest.mark.parametrize("mode", [1, 2])
def test_obdeect_chain_rejects_atmosphere_applied_twice(fake_atmosphere_and_optics, mode):
    spec = ChainSpec(_OBDEECT_CHAIN, {"sim_telarray": {"bypass_optics": mode}})

    with pytest.raises(ValueError, match="applies 'atmospheric_extinction' twice"):
        spec.chain()
