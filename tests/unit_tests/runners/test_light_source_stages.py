#!/usr/bin/python3

import shlex
from pathlib import Path

import astropy.units as u
import pytest

from simtools.runners import stages
from simtools.runners.chain_runner import ChainRunner
from simtools.runners.light_source_stages import (
    FlasherSourceStage,
    IlluminatorSourceStage,
)
from simtools.runners.stages import StageInputs, StreamTo
from simtools.simulation_chain import light_sources
from simtools.simulation_chain.products import (
    CAMERA_EVENTS,
    PHOTONS_CAMERA,
    PHOTONS_SKY,
)

_FLASHER = "LSFN-design"
_ILLUMINATOR = "ILLN-01"


def _record(value, unit=None):
    return {
        "value": value,
        "unit": unit,
        "type": "float64",
        "file": False,
        "parameter_version": "1.0.0",
        "model_parameter_schema_version": "1.0.0",
    }


_TELESCOPE_PARAMETERS = {
    "mirror_offset": _record(50.0, "cm"),
    "camera_body_diameter": _record(240.0, "cm"),
    "camera_body_shape": _record(0),
    "axes_offsets": _record([0.0, 0.0], ["cm", "cm"]),
}
_FLASHER_PARAMETERS = {
    "flasher_position": _record([2.0, 0.0, 0.0], ["cm", "cm", "cm"]),
    "flasher_wavelength": _record(355.0, "nm"),
    "flasher_photons": _record(1000),
    "flasher_bunch_size": _record(1),
    "flasher_type": _record("flat_fielding"),
    "flasher_angular_distribution": _record("FilledCone"),
    "flasher_angular_distribution_width": _record(2.5, "deg"),
    "flasher_pulse_shape": _record(["Gauss", 1.5, 0.0], [None, "ns", "ns"]),
}
_ILLUMINATOR_PARAMETERS = _FLASHER_PARAMETERS | {
    "flasher_type": _record("illuminator"),
    "flasher_angular_distribution": _record("isotropic"),
    "flasher_angular_distribution_width": _record(0.0, "deg"),
    "array_element_position_ground": _record([110.0, 0.0, 0.0], "m"),
    "illuminator_tower_height": _record(5.0, "m"),
}


@pytest.fixture
def light_source_context(run_context, mock_model_reader):
    """Run context whose model reader also provides calibration devices."""
    original = mock_model_reader.get_model_parameters.side_effect

    def get_model_parameters(site, array_element_name, collection, model_version, **kwargs):
        parameters = original(site, array_element_name, collection, model_version, **kwargs)
        if collection == "calibration_devices":
            extra = _ILLUMINATOR_PARAMETERS if "ILL" in array_element_name else _FLASHER_PARAMETERS
            return parameters | extra
        return parameters | _TELESCOPE_PARAMETERS

    mock_model_reader.get_model_parameters.side_effect = get_model_parameters
    return run_context.replace(args={"simulation_seed": [3, 21]})


def _inputs(context, **args):
    base = {
        "site": "North",
        "model_version": "6.0.2",
        "telescope": "LSTN-01",
        "number_of_events": 2,
        "run_number": 1,
    }
    return StageInputs(
        array_models=(),
        simulation_parameters=(),
        shower_configurations=(),
        args=base | args,
        context=context,
    )


def test_registered_stage_classes():
    assert stages.get_stage_class("obdeect_flasher") is FlasherSourceStage
    assert stages.get_stage_class("obdeect_illuminator") is IlluminatorSourceStage
    for stage in (FlasherSourceStage, IlluminatorSourceStage):
        assert stage.consumes is None
        assert stage.streams is True
        assert stage.applied_factors({}) == frozenset()
    assert FlasherSourceStage.produces == PHOTONS_CAMERA
    assert IlluminatorSourceStage.produces == PHOTONS_SKY


def test_flasher_stage_instance(light_source_context):
    inputs = _inputs(light_source_context, light_source=_FLASHER)
    (stage,) = FlasherSourceStage.instances(inputs, label="test")
    assert isinstance(stage.request, light_sources.FlasherRequest)
    assert stage.request.events == (2,)
    assert stage.request.photons == (1000,)
    assert stage.request.wavelength_nm == pytest.approx(355.0)
    assert stage.request.seed == 21  # Simulation seed.
    assert stage.file_token == "_LSTN-01_LSFN-design_355nm"
    assert stage.get_resources(runtime=3.0) == {"runtime": 3.0, "n_events": 2}
    assert stage.environment() == {}


def test_stage_options_take_precedence(light_source_context):
    inputs = _inputs(light_source_context, light_source=_FLASHER, number_of_events=2)
    inputs = StageInputs(
        **{
            **inputs.__dict__,
            "chain_options": {"obdeect_flasher": {"number_of_events": 5, "wavelength": 355}},
        }
    )
    (stage,) = FlasherSourceStage.instances(inputs)
    assert stage.request.events == (5,)
    assert stage.request.wavelength_nm == pytest.approx(355.0)


def test_illuminator_stage_instance(light_source_context):
    inputs = _inputs(
        light_source_context,
        light_source=_ILLUMINATOR,
        wavelength=[355.0 * u.nm],
        light_source_position=[0.0, 0.0, 100.0],
        light_source_pointing=[0.0, 0.0, -1.0],
    )
    (stage,) = IlluminatorSourceStage.instances(inputs)
    assert isinstance(stage.request, light_sources.IlluminatorRequest)
    assert stage.request.position_m == pytest.approx((0.0, 0.0, 100.0))
    assert stage.request.events == 2


def test_stage_rejects_other_light_source_type(light_source_context):
    inputs = _inputs(light_source_context, light_source=_ILLUMINATOR, wavelength=355.0 * u.nm)
    with pytest.raises(ValueError, match="is of type 'illuminator'"):
        FlasherSourceStage.instances(inputs)


def test_stage_needs_one_telescope(light_source_context):
    inputs = _inputs(
        light_source_context,
        light_source=_FLASHER,
        telescope=None,
        telescopes=["LSTN-01", "LSTN-02"],
    )
    with pytest.raises(ValueError, match="exactly one telescope"):
        FlasherSourceStage.instances(inputs)
    inputs = _inputs(light_source_context, light_source=_FLASHER, telescope=None, telescopes=None)
    with pytest.raises(ValueError, match="exactly one telescope"):
        FlasherSourceStage.instances(inputs)


def test_stage_accepts_single_telescope_list(light_source_context):
    inputs = _inputs(
        light_source_context, light_source=_FLASHER, telescope=None, telescopes=["LSTN-01"]
    )
    assert FlasherSourceStage.instances(inputs)[0].file_token.startswith("_LSTN-01")


@pytest.fixture
def flasher_stage(light_source_context):
    inputs = _inputs(light_source_context, light_source=_FLASHER)
    return FlasherSourceStage.instances(inputs)[0]


def test_prepare_writes_file_without_stream(flasher_stage):
    (line,) = flasher_stage.prepare(5)
    tokens = shlex.split(line)
    assert tokens[1:3] == ["-m", "simtools.simulation_chain.flasher_stream"]
    request = light_sources.flasher_request_from_json(tokens[tokens.index("--request") + 1])
    assert request.run_number == 5
    assert (request.telescope, request.site, request.model_version) == (
        "LSTN-01",
        "North",
        "6.0.2",
    )
    assert request.instrument_seed == 3
    output = flasher_stage.output_file(5)
    assert tokens[tokens.index("--output") + 1] == str(output)
    assert output.name == "obdeect_flasher_LSTN-01_LSFN-design_355nm_run000005.pst"


def test_prepare_streams_to_downstream_command(flasher_stage):
    (line,) = flasher_stage.prepare(1, output=StreamTo(("obdeect-stream-to-simtel -",)))
    command, downstream = line.split(" | ")
    assert "--output" not in shlex.split(command)
    assert downstream == "obdeect-stream-to-simtel -"


def test_prepare_errors(flasher_stage):
    with pytest.raises(ValueError, match="does not read an input"):
        flasher_stage.prepare(1, input_file="in.pst")
    with pytest.raises(ValueError, match="exactly one downstream command"):
        flasher_stage.prepare(1, output=StreamTo(("a", "b")))


def test_illuminator_prepare_writes_position_file(light_source_context):
    inputs = _inputs(light_source_context, light_source=_ILLUMINATOR, wavelength=355.0 * u.nm)
    (stage,) = IlluminatorSourceStage.instances(inputs)
    tokens = shlex.split(stage.prepare(2)[0])
    assert tokens[0] == "obdeect-sources-point-source"
    position_file = tokens[tokens.index("--telpos-file") + 1]
    assert len(Path(position_file).read_text(encoding="utf-8").split()) == 4


class _FakeDownstream:
    """Downstream stage consuming a product and streaming."""

    consumes = PHOTONS_CAMERA
    produces = CAMERA_EVENTS
    streams = True
    name = "fake_downstream"
    file_list = None

    @classmethod
    def instances(cls, inputs, label=None):
        return [cls()]

    def prepare(self, run_number, input_file=None, output=None):
        return [f"fake-downstream {input_file}"]

    def environment(self):
        return {}


def test_chain_with_fake_downstream_stage(monkeypatch, light_source_context):
    monkeypatch.setattr(stages, "STAGES", dict(stages.STAGES))
    stages.register_stage("fake_downstream", lambda: _FakeDownstream)
    chain = stages.SimulationChain(("obdeect_flasher", "fake_downstream"))
    assert chain.starts_with_showers
    runner = ChainRunner(chain, _inputs(light_source_context, light_source=_FLASHER))
    script = runner.stages[0][0].work_directory / "run.sh"
    runner.prepare_run(1, script)
    lines = script.read_text(encoding="utf-8").splitlines()
    assert "simtools.simulation_chain.flasher_stream" in lines[-1]
    assert lines[-1].endswith(" | fake-downstream -")


def test_chain_rejects_mismatching_products():
    with pytest.raises(ValueError, match="consumes None"):
        stages.SimulationChain(("obdeect_illuminator", "obdeect_flasher"))
