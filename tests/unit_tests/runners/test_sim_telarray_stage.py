"""Tests for the sim_telarray simulation stage."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from simtools.runners import stages
from simtools.runners.chain_spec import (
    ATMOSPHERIC_EXTINCTION,
    MIRROR_REFLECTIVITY,
    QUANTUM_EFFICIENCY,
    TELESCOPE_TRANSMISSION,
    ChainSpec,
)
from simtools.runners.sim_telarray_stage import LightSimTelArrayStage, SimTelArrayStage
from simtools.runners.stages import STDIN, StageInputs, StreamTo
from simtools.simulation_chain.products import CAMERA_PHOTONS_IACT, CHERENKOV_PHOTONS

ALL_FACTORS = frozenset(
    {ATMOSPHERIC_EXTINCTION, MIRROR_REFLECTIVITY, TELESCOPE_TRANSMISSION, QUANTUM_EFFICIENCY}
)


class _Telescope:
    """Telescope model with the mirror class of the bypass level."""

    def __init__(self, mirror_class):
        self._mirror_class = mirror_class

    def get_parameter_value(self, name):
        assert name == "mirror_class"
        return self._mirror_class


@pytest.fixture
def sim_telarray_stage(corsika_config_mock_array_model):
    return SimTelArrayStage(
        simulation_parameters=corsika_config_mock_array_model.simulation_parameters,
        array_model=corsika_config_mock_array_model.array_model,
        label="test-sim-telarray-stage",
        context=corsika_config_mock_array_model.context,
    )


def test_instances_one_per_array_model(corsika_config_mock_array_model):
    inputs = StageInputs(
        array_models=(corsika_config_mock_array_model.array_model,) * 2,
        simulation_parameters=(corsika_config_mock_array_model.simulation_parameters,) * 2,
        shower_configurations=(),
        args={},
        context=corsika_config_mock_array_model.context,
    )

    assert len(SimTelArrayStage.instances(inputs, label="test")) == 2


def test_prepare_with_input_file(sim_telarray_stage, mocker):
    mocker.patch.object(
        sim_telarray_stage.simulator_array,
        "make_run_command",
        return_value=["sim_telarray", "command"],
    )
    sim_telarray_stage.simulator_array.file_list = {}

    lines = sim_telarray_stage.prepare(run_number=1, input_file="input.corsika")

    assert len(lines) == 1
    assert "sim_telarray command" in lines[0]
    assert "input.corsika" in lines[0]
    assert "SIM_TELARRAY_CONFIG_PATH" in lines[0]


def test_prepare_keeps_resource_record_for_each_run(sim_telarray_stage, mocker):
    mocker.patch.object(
        sim_telarray_stage.simulator_array,
        "make_run_command",
        return_value=["sim_telarray", "command"],
    )
    sim_telarray_stage.simulator_array.file_list = {}
    sim_telarray_stage.simulator_array.runs_per_set = 2

    lines = sim_telarray_stage.prepare(run_number=1, input_file="input.corsika")

    resource_files = sim_telarray_stage.file_list["sim_telarray_resources"]
    assert len(lines) == 2
    assert resource_files[0] != resource_files[1]
    assert str(resource_files[0]) in lines[0]
    assert str(resource_files[1]) in lines[1]


def test_prepare_reading_stream(sim_telarray_stage):
    lines = sim_telarray_stage.prepare(run_number=1, input_file=STDIN)

    assert len(lines) == 1
    assert "bin/sim_telarray" in lines[0]
    assert "-C show=all" in lines[0]
    assert "--input-file" not in lines[0]


def test_prepare_reading_stream_lists_resources(sim_telarray_stage):
    lines = sim_telarray_stage.prepare(run_number=1, input_file=STDIN)

    resources = sim_telarray_stage.file_list["sim_telarray_resources"]
    assert resources == sim_telarray_stage.runner_service.get_file_name(
        file_type="sim_telarray_resources", run_number=1
    )
    assert str(resources) in lines[0]


def test_prepare_rejects_streamed_output(sim_telarray_stage):
    with pytest.raises(ValueError, match="sim_telarray output can only be written"):
        sim_telarray_stage.prepare(run_number=1, input_file=STDIN, output=StreamTo(("x",)))


def test_output_file_environment_resources(sim_telarray_stage):
    assert "sim_telarray" in str(sim_telarray_stage.output_file(1))
    assert sim_telarray_stage.environment() == {"SIM_TELARRAY_CONFIG_PATH": ""}
    assert sim_telarray_stage.get_resources(runtime=1.5)["runtime"] == pytest.approx(1.5)


def test_environment_includes_photoelectron_override(sim_telarray_stage):
    simulator = sim_telarray_stage.simulator_array
    simulator.context = simulator.context.replace(args={"max_photoelectrons": 12345})
    assert sim_telarray_stage.environment() == {
        "SIM_TELARRAY_CONFIG_PATH": "",
        "MAX_PHOTOELECTRONS": "12345",
    }


@pytest.mark.parametrize("mode", ["stream", "disk"])
def test_instances_configure_shower_routing(corsika_config_mock_array_model, mode):
    from dataclasses import replace

    config = corsika_config_mock_array_model
    context = config.context.replace(
        software=replace(config.context.software, ray_tracing_backend="obdeect")
    )
    inputs = StageInputs(
        array_models=(config.array_model,),
        simulation_parameters=(config.simulation_parameters,),
        shower_configurations=(config,),
        args={"shower_route_mode": mode},
        context=context,
    )
    (stage,) = SimTelArrayStage.instances(inputs)
    assert stage.shower_routing is not None
    assert stage.streams == (mode == "stream")


def test_instances_reject_invalid_shower_route_mode(corsika_config_mock_array_model):
    from dataclasses import replace

    config = corsika_config_mock_array_model
    context = config.context.replace(
        software=replace(config.context.software, ray_tracing_backend="obdeect")
    )
    inputs = StageInputs(
        array_models=(config.array_model,),
        simulation_parameters=(config.simulation_parameters,),
        shower_configurations=(config,),
        args={"shower_route_mode": "file"},
        context=context,
    )
    with pytest.raises(ValueError, match="Invalid shower route mode"):
        SimTelArrayStage.instances(inputs)


def test_consumes_corsika_or_camera_photons(sim_telarray_stage):
    assert SimTelArrayStage.consumes == (CHERENKOV_PHOTONS, CAMERA_PHOTONS_IACT)
    assert sim_telarray_stage.bypass_optics is None


def test_prepare_without_options_has_no_bypass_option(sim_telarray_stage):
    lines = sim_telarray_stage.prepare(run_number=1, input_file=STDIN)

    assert "Bypass_Optics" not in lines[0]


def test_prepare_sets_bypass_optics_before_show_all(corsika_config_mock_array_model):
    config = corsika_config_mock_array_model
    stage = SimTelArrayStage(
        simulation_parameters=config.simulation_parameters,
        array_model=config.array_model,
        label="test-bypass",
        context=config.context,
        bypass_optics=2,
    )

    lines = stage.prepare(run_number=1, input_file=STDIN)

    assert "-C Bypass_Optics=2 -C show=all -" in lines[0]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, ALL_FACTORS),
        ({"bypass_optics": 0}, ALL_FACTORS),
        ({"bypass_optics": 1}, {QUANTUM_EFFICIENCY, ATMOSPHERIC_EXTINCTION}),
        ({"bypass_optics": 2}, {QUANTUM_EFFICIENCY, ATMOSPHERIC_EXTINCTION}),
        ({"bypass_optics": 3}, {QUANTUM_EFFICIENCY}),
    ],
)
def test_applied_factors_by_bypass_mode(options, expected):
    assert SimTelArrayStage.applied_factors(options) == frozenset(expected)


def test_applied_factors_rejects_unknown_mode():
    with pytest.raises(ValueError, match=r"Invalid bypass_optics option: 4"):
        SimTelArrayStage.applied_factors({"bypass_optics": 4})


def test_neutral_atmosphere_option_is_rejected():
    with pytest.raises(
        ValueError, match=r"Unknown sim_telarray stage option\(s\): neutral_atmosphere"
    ):
        SimTelArrayStage.applied_factors({"bypass_optics": 2, "neutral_atmosphere": True})


def _release_output(release):
    """Return a fake result of ``sim_telarray --version`` printing a release."""
    return SimpleNamespace(stdout=f"Release: {release} from 2024-09-27\n", stderr="")


def _bypass_inputs(config, options):
    return StageInputs(
        array_models=(config.array_model,),
        simulation_parameters=(config.simulation_parameters,),
        shower_configurations=(),
        args={},
        context=config.context,
        chain_options={"sim_telarray": options},
    )


def test_chain_without_bypass_mode_3_does_not_probe_sim_telarray(
    corsika_config_mock_array_model, mocker
):
    run = mocker.patch(
        "simtools.runtime.dependencies.subprocess.run", return_value=_release_output("2024.271.0")
    )
    config = corsika_config_mock_array_model
    config.array_model.telescope_models = {"tel-0": _Telescope(2)}

    (stage,) = SimTelArrayStage.instances(_bypass_inputs(config, {}))
    SimTelArrayStage.instances(_bypass_inputs(config, {"bypass_optics": 2}))

    run.assert_not_called()
    assert stage.bypass_optics is None


def test_mode_3_refused_by_installed_release(corsika_config_mock_array_model, mocker, monkeypatch):
    monkeypatch.delenv("SIMTOOLS_SIM_TELARRAY_BYPASS_OPTICS_MAX", raising=False)
    run = mocker.patch(
        "simtools.runtime.dependencies.subprocess.run", return_value=_release_output("2024.271.0")
    )
    config = corsika_config_mock_array_model

    with pytest.raises(ValueError, match=r"Bypass_Optics=3 is not supported.*2024\.271\.0") as exc:
        SimTelArrayStage.instances(_bypass_inputs(config, {"bypass_optics": 3}))

    assert "SIMTOOLS_SIM_TELARRAY_BYPASS_OPTICS_MAX" in str(exc.value)
    run.assert_called_once()


def test_mode_3_accepted_with_override_without_probe(
    corsika_config_mock_array_model, mocker, monkeypatch
):
    monkeypatch.setenv("SIMTOOLS_SIM_TELARRAY_BYPASS_OPTICS_MAX", "3")
    run = mocker.patch("simtools.runtime.dependencies.subprocess.run")
    config = corsika_config_mock_array_model

    (stage,) = SimTelArrayStage.instances(_bypass_inputs(config, {"bypass_optics": 3}))

    run.assert_not_called()
    assert stage.bypass_optics == 3


def test_mode_3_accepted_by_release_table(corsika_config_mock_array_model, mocker, monkeypatch):
    monkeypatch.delenv("SIMTOOLS_SIM_TELARRAY_BYPASS_OPTICS_MAX", raising=False)
    monkeypatch.setattr(
        "simtools.simtel.sim_telarray_capabilities.BYPASS_OPTICS_MODE_3_MIN_VERSION", "2026.300.0"
    )
    mocker.patch(
        "simtools.runtime.dependencies.subprocess.run", return_value=_release_output("2026.301.0")
    )
    config = corsika_config_mock_array_model

    (stage,) = SimTelArrayStage.instances(_bypass_inputs(config, {"bypass_optics": 3}))

    assert stage.bypass_optics == 3


@pytest.mark.parametrize(
    ("mode", "telescopes", "expected"),
    [
        (0, (1, 2), None),
        (1, (1, 1), 1),
        (2, (2, 2), 2),
        (3, (1, 1), 3),
    ],
)
def test_instances_bypass_level_from_mode(
    corsika_config_mock_array_model, mode, telescopes, expected, monkeypatch
):
    # Mode 3 is gated by the sim_telarray capability; the override avoids running sim_telarray.
    monkeypatch.setenv("SIMTOOLS_SIM_TELARRAY_BYPASS_OPTICS_MAX", "3")
    config = corsika_config_mock_array_model
    config.array_model.telescope_models = {
        f"tel-{index}": _Telescope(mirror_class) for index, mirror_class in enumerate(telescopes)
    }
    inputs = StageInputs(
        array_models=(config.array_model,),
        simulation_parameters=(config.simulation_parameters,),
        shower_configurations=(),
        args={},
        context=config.context,
        chain_options={"sim_telarray": {"bypass_optics": mode}},
    )

    (stage,) = SimTelArrayStage.instances(inputs)

    assert stage.bypass_optics == expected


def test_instances_bypass_rejects_mixed_mirror_classes(corsika_config_mock_array_model):
    config = corsika_config_mock_array_model
    config.array_model.telescope_models = {"single": _Telescope(1), "dual": _Telescope(2)}
    inputs = StageInputs(
        array_models=(config.array_model,),
        simulation_parameters=(config.simulation_parameters,),
        shower_configurations=(),
        args={},
        context=config.context,
        chain_options={"sim_telarray": {"bypass_optics": 1}},
    )

    with pytest.raises(ValueError, match="needs telescopes of one mirror class"):
        SimTelArrayStage.instances(inputs)


def test_instances_reject_invalid_bypass_option(corsika_config_mock_array_model):
    config = corsika_config_mock_array_model
    inputs = StageInputs(
        array_models=(config.array_model,),
        simulation_parameters=(config.simulation_parameters,),
        shower_configurations=(),
        args={},
        context=config.context,
        chain_options={"sim_telarray": {"bypass_optics": "yes"}},
    )

    with pytest.raises(ValueError, match="Invalid bypass_optics option"):
        SimTelArrayStage.instances(inputs)


@pytest.fixture
def fake_photon_stage(monkeypatch):
    """Register a stage that applies quantum efficiency upstream of sim_telarray."""

    class FakeStage:
        name = "fake_qe"
        consumes = None
        produces = CHERENKOV_PHOTONS
        streams = True

        @classmethod
        def applied_factors(cls, options):
            return frozenset({QUANTUM_EFFICIENCY})

    monkeypatch.setitem(stages.STAGES, "fake_qe", lambda: FakeStage)


def test_chain_rejects_quantum_efficiency_applied_twice(fake_photon_stage):
    spec = ChainSpec(("fake_qe", "sim_telarray"))

    with pytest.raises(ValueError, match="applies 'quantum_efficiency' twice"):
        spec.chain()


def _light_inputs(light_simulation, stage_names, options=None):
    return StageInputs(
        array_models=(),
        simulation_parameters=(),
        shower_configurations=(),
        args={},
        context=light_simulation.context,
        chain_options={"sim_telarray": options} if options else {},
        stages=tuple(stage_names),
        calibration=light_simulation,
    )


def test_light_instances_replace_array_stages(light_simulation):
    inputs = _light_inputs(light_simulation, ("light_emission", "sim_telarray"))
    (stage,) = SimTelArrayStage.instances(inputs)
    assert isinstance(stage, LightSimTelArrayStage)
    assert stage.bypass_optics is None
    assert stage.consumes == SimTelArrayStage.consumes
    assert stage.streams is False


@pytest.mark.parametrize(
    ("mirror_class", "option", "level"), [(0, 1, 1), (2, 1, 2), (2, 2, 2), (0, 0, 0)]
)
def test_light_instances_bypass_level_from_option(light_simulation, mirror_class, option, level):
    light_simulation.telescope_model.get_parameter_value.return_value = mirror_class
    inputs = _light_inputs(
        light_simulation,
        ("obdeect_flasher", "stream_to_simtel", "sim_telarray"),
        {"bypass_optics": option},
    )
    (stage,) = SimTelArrayStage.instances(inputs)
    assert stage.bypass_optics == level


def test_light_instances_bypass_level_three_checks_capability(light_simulation):
    inputs = _light_inputs(
        light_simulation,
        ("obdeect_illuminator", "stream_to_simtel", "sim_telarray"),
        {"bypass_optics": 3},
    )
    with patch("simtools.runners.sim_telarray_stage.check_bypass_optics") as check:
        (stage,) = SimTelArrayStage.instances(inputs)
    check.assert_called_once_with(3, light_simulation.context)
    assert stage.bypass_optics == 3


def test_light_instances_camera_photons_need_bypass_option(light_simulation):
    inputs = _light_inputs(
        light_simulation, ("obdeect_flasher", "stream_to_simtel", "sim_telarray")
    )
    with pytest.raises(ValueError, match="option 'bypass_optics' of stage 'sim_telarray' is req"):
        SimTelArrayStage.instances(inputs)


def test_light_instances_reject_unknown_option(light_simulation):
    inputs = _light_inputs(light_simulation, ("light_emission", "sim_telarray"), {"other": 1})
    with pytest.raises(ValueError, match="Unknown sim_telarray stage option"):
        SimTelArrayStage.instances(inputs)


@pytest.mark.parametrize("light_source_type", ["flat_fielding", "illuminator"])
def test_light_stage_command_equals_light_emission_script(light_simulation, light_source_type):
    light_simulation.light_emission_config["light_source_type"] = light_source_type
    inputs = _light_inputs(light_simulation, ("light_emission", "sim_telarray"))
    (stage,) = SimTelArrayStage.instances(inputs)
    iact_file = light_simulation.runner_service.get_file_name("iact_output")
    with patch.object(light_simulation, "_get_telescope_pointing", return_value=(0.0, 0.0)):
        lines = stage.prepare(3, input_file=iact_file)
        script = light_simulation._make_simtel_script()
    assert lines == [script.rstrip("\n"), f"rm -f '{iact_file}'"]
    assert (" Bypass_Optics=1" in script) == (light_source_type == "flat_fielding")
    assert stage.output_file(3) == light_simulation.runner_service.get_file_name(
        "sim_telarray_output"
    )


def test_light_stage_uses_input_file_of_upstream_stage(light_simulation):
    stage = LightSimTelArrayStage(light_simulation, bypass_optics=3)
    (line,) = stage.prepare(1, input_file=Path("/in/camera.iact"))
    assert "-C input_file=/in/camera.iact" in line
    assert "-C Bypass_Optics=3" in line
    assert "rm -f" not in line


def test_light_stage_errors(light_simulation):
    stage = LightSimTelArrayStage(light_simulation)
    with pytest.raises(ValueError, match="reads a file and writes its file"):
        stage.prepare(1, input_file=STDIN)
    with pytest.raises(ValueError, match="reads a file and writes its file"):
        stage.prepare(1, output=StreamTo(("downstream",)))
    assert stage.environment() == {}
    assert stage.get_resources(runtime=1.5) == {"runtime": 1.5}
