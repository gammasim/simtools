#!/usr/bin/python3

import dataclasses
import json
import math
import shlex
from pathlib import Path

import astropy.units as u
import pytest

from simtools.runners import adapter_stages, light_emission_runner, stages
from simtools.runners.chain_runner import ChainRunner
from simtools.runners.chain_spec import SimulationChain, parse_chain
from simtools.runners.light_source_stages import LightSourceStage
from simtools.runners.stages import StageInputs, StreamTo, get_stage_class
from simtools.simulation_software import light_sources

LIGHT_EMISSION_PHOTONS = "light_emission_photons"
PHOTONS_CAMERA = "photons_camera"
PHOTONS_SKY = "photons_sky"
FlasherSourceStage = get_stage_class("obdeect_flasher")
IlluminatorSourceStage = get_stage_class("obdeect_illuminator")
LightEmissionStage = get_stage_class("light_emission")


class _RunFiles:
    """File naming of a run: ``<directory>/run<N>.<stage>.pst``."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def get_stage_file_name(self, stage_name, file_type, run_number=None):
        assert file_type == "photon_stream"
        return self.directory / f"run{run_number:06d}.{stage_name}.pst"


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
        calibration=type(
            "Calibration",
            (),
            {"runner_service": _RunFiles(context.paths.get_output_directory("light"))},
        ),
    )


def test_registered_stage_classes():
    for stage in (FlasherSourceStage, IlluminatorSourceStage):
        assert issubclass(stage, LightSourceStage)
        assert stage.consumes is None
        assert stage.streams is True
        assert stages.applied_factors_of(stage.name, {}) == frozenset()
    assert (FlasherSourceStage.spec.light_source, IlluminatorSourceStage.spec.light_source) == (
        "flasher",
        "illuminator",
    )
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
    chain = parse_chain(["obdeect_flasher:number_of_events=5,wavelength=355"])
    inputs = dataclasses.replace(inputs, chain=chain)
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
    assert tokens[1:3] == ["-m", "simtools.simulation_software.flasher_stream"]
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
    assert output.name == "run000005.obdeect_flasher.pst"


def test_prepare_streams_to_downstream_command(flasher_stage):
    (line,) = flasher_stage.prepare(1, output=StreamTo(("obdeect-stream-to-simtel -",)))
    command, downstream = line.split(" | ")
    assert "--output" not in shlex.split(command)
    assert downstream == "obdeect-stream-to-simtel -"


def test_prepare_errors(flasher_stage):
    with pytest.raises(ValueError, match="does not read an input"):
        flasher_stage.prepare(1, input_file="in.pst")
    with pytest.raises(ValueError, match="pipes into one downstream command, got 2"):
        flasher_stage.prepare(1, output=StreamTo(("a", "b")))


def test_illuminator_prepare_writes_position_file(light_source_context):
    inputs = _inputs(light_source_context, light_source=_ILLUMINATOR, wavelength=355.0 * u.nm)
    (stage,) = IlluminatorSourceStage.instances(inputs)
    tokens = shlex.split(stage.prepare(2)[0])
    assert tokens[0].endswith("obdeect-sources-point-source")
    position_file = tokens[tokens.index("--telpos-file") + 1]
    assert len(Path(position_file).read_text(encoding="utf-8").split()) == 4


class _FakeDownstream:
    """Downstream stage consuming a product and streaming."""

    consumes = PHOTONS_CAMERA
    produces = "fake_product"
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
    monkeypatch.setitem(stages.registry(), "fake_downstream", _FakeDownstream)
    chain = SimulationChain(("obdeect_flasher", "fake_downstream"))
    assert chain.source == stages.LIGHT_SOURCE
    runner = ChainRunner(chain, _inputs(light_source_context, light_source=_FLASHER))
    script = runner.stages[0][0].work_directory / "run.sh"
    runner.prepare_run(1, script)
    lines = script.read_text(encoding="utf-8").splitlines()
    assert "simtools.simulation_software.flasher_stream" in lines[-1]
    assert lines[-1].endswith(" | fake-downstream -")


def test_chain_rejects_mismatching_products():
    with pytest.raises(ValueError, match="consumes None"):
        SimulationChain(("obdeect_illuminator", "obdeect_flasher"))


def _calibration_inputs(light_simulation):
    return StageInputs(
        array_models=(),
        simulation_parameters=(),
        shower_configurations=(),
        args={},
        context=light_simulation.context,
        calibration=light_simulation,
    )


def test_light_emission_stage_registered():
    assert stages.stages_software(["light_emission"]) == {"sim_telarray"}
    assert LightEmissionStage.consumes is None
    assert LightEmissionStage.produces == LIGHT_EMISSION_PHOTONS
    assert LightEmissionStage.streams is False
    assert stages.applied_factors_of("light_emission", {}) == frozenset()


def test_light_emission_stage_lines_are_those_of_the_run_command(light_simulation):
    (stage,) = LightEmissionStage.instances(_calibration_inputs(light_simulation))
    lines = stage.prepare(3)
    run_command = light_simulation.make_run_command()
    assert [f"{line}\n\n" for line in lines] == run_command[2:4]
    assert stage.output_file(3) == light_simulation.runner_service.get_file_name("iact_output")
    assert stage.environment() == {}
    assert stage.get_resources(runtime=2.0) == {"runtime": 2.0}


def test_light_emission_stage_needs_light_emission_simulation(run_context):
    inputs = StageInputs(
        array_models=(), simulation_parameters=(), shower_configurations=(), args={}
    )
    with pytest.raises(ValueError, match="needs a light-emission simulation"):
        LightEmissionStage.instances(inputs)


def test_light_emission_stage_cannot_stream_or_read(light_simulation):
    (stage,) = LightEmissionStage.instances(_calibration_inputs(light_simulation))
    with pytest.raises(ValueError, match="reads no input"):
        stage.prepare(1, input_file="photons.iact")
    with pytest.raises(ValueError, match="reads no input"):
        stage.prepare(1, output=StreamTo(("downstream",)))


def test_flasher_chain_through_light_emission_simulation(
    light_simulation, light_source_context, tmp_test_directory, monkeypatch
):
    monkeypatch.setattr(
        adapter_stages,
        "export_atmosphere_files",
        lambda _site, directory: (Path(directory) / "atmosphere-profile.dat", None),
    )
    light_simulation.context = light_source_context
    light_simulation.light_emission_config |= {
        "site": "North",
        "model_version": "6.0.2",
        "light_source": _FLASHER,
        "number_of_events": 2,
        "telescope": "LSTN-01",
    }
    light_simulation.telescope = "LSTN-01"
    light_simulation.telescope_model.get_parameter_value_with_unit.side_effect = lambda name: (
        0.0 * u.deg if name == "camera_rotate" else 2156.0 * u.m
    )
    script = tmp_test_directory / "chain.sh"
    items = ["obdeect_flasher", "stream_to_simtel", "sim_telarray:bypass_optics=camera_entrance"]
    light_emission_runner.write_chain_script(light_simulation, items, script)
    lines = script.read_text(encoding="utf-8").splitlines()
    camera_iact = tmp_test_directory / "run.stream_to_simtel.photon_bunches"
    assert "simtools.simulation_software.flasher_stream" in lines[3]
    assert " | " in lines[3]
    assert lines[3].endswith(f"> {camera_iact}")
    camera_map = tmp_test_directory / "run.stream_to_simtel.camera_map"
    assert camera_map.read_text(encoding="utf-8") == "1 1 0.0\n"
    assert "--atmprof " in lines[3]
    assert f"-C input_file={camera_iact}" in lines[4]
    assert "-C Bypass_Optics=1" in lines[4]
    assert len(lines) == 5


def test_flasher_worker_receives_resolved_model_source(light_source_context):
    source = {"type": "git", "repository": "https://example.test/models.git", "commit": "a" * 40}
    light_source_context.model_reader.source_config = source
    context = light_source_context.replace(
        args={**light_source_context.args, "overwrite_model_parameters": {"LSTN-01": {}}}
    )
    (stage,) = FlasherSourceStage.instances(_inputs(context, light_source=_FLASHER))
    tokens = shlex.split(stage.prepare(1)[0])
    environment = next(token for token in tokens if token.startswith("SIMTOOLS_FLASHER_CONTEXT="))
    payload = json.loads(Path(environment.split("=", 1)[1]).read_text(encoding="utf-8"))
    assert payload["model_source"] == source
    assert payload["overwrite_model_parameters"] == {"LSTN-01": {}}


def test_worker_model_uses_parent_overrides(light_source_context):
    from simtools.simulation_software.flasher_stream import create_telescope_model

    (stage,) = FlasherSourceStage.instances(_inputs(light_source_context, light_source=_FLASHER))
    overrides = {"LSTN-01": {"mirror_offset": {"value": 65.0, "unit": "cm"}}}
    context = light_source_context.replace(args={"overwrite_model_parameters": overrides})
    telescope = create_telescope_model(stage.request, context=context)
    assert math.isclose(telescope.get_parameter_value("mirror_offset"), 65.0)
