"""Tests for the sim_telarray simulation stage."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from astropy.table import Column, Table

from simtools.runners.chain_spec import parse_chain
from simtools.runners.sim_telarray_stage import LightSimTelArrayStage, _bypass_level
from simtools.runners.stages import STDIN, StageInputs, StreamTo, get_stage_class

CAMERA_PHOTONS_IACT = "camera_photons_iact"
CHERENKOV_PHOTONS = "cherenkov_photons"
LIGHT_EMISSION_PHOTONS = "light_emission_photons"
ATMOSPHERIC_EXTINCTION = "atmospheric_extinction"
MIRROR_REFLECTIVITY = "mirror_reflectivity"
QUANTUM_EFFICIENCY = "quantum_efficiency"
TELESCOPE_TRANSMISSION = "telescope_transmission"
SimTelArrayStage = get_stage_class("sim_telarray")

ALL_FACTORS = frozenset(
    {ATMOSPHERIC_EXTINCTION, MIRROR_REFLECTIVITY, TELESCOPE_TRANSMISSION, QUANTUM_EFFICIENCY}
)
#: Air-shower chains by bypass mode (the photons of the source reach sim_telarray through
#: the atmosphere and optics stages unless sim_telarray applies its own optics).
CORSIKA_OPTICS_SIM_TELARRAY = ("corsika", "sim_telarray")
CAMERA_ONLY_SHOWER = (
    "corsika",
    "obdeect_atmosphere",
    "obdeect_optics",
    "sim_telarray:bypass_optics=camera_only",
)
CAMERA_ENTRANCE_SHOWER = (
    "corsika",
    "obdeect_atmosphere",
    "obdeect_optics",
    "sim_telarray:bypass_optics=camera_entrance",
)
CAMERA_ENTRANCE_NO_EXTINCTION = (
    "corsika",
    "obdeect_optics",
    "sim_telarray:bypass_optics=camera_entrance",
)


class _Telescope:
    """Telescope model with the mirror class of the bypass level."""

    def __init__(self, mirror_class):
        self._mirror_class = mirror_class

    def get_parameter_value(self, name):
        assert name == "mirror_class"
        return self._mirror_class


class _EnvelopeTelescope:
    """Telescope model with a QE table, the parts of a model the envelope reads."""

    def __init__(self, name):
        self.name = name
        self.parameters = {"quantum_efficiency": {"value": "qe.ecsv", "instrument": name}}
        self._table = Table(
            {
                "wavelength": Column([300.0, 500.0], unit="nm"),
                "efficiency": [0.2, 0.4],
            }
        )

    def get_parameter_table(self, parameter_name):
        assert parameter_name == "quantum_efficiency"
        return self._table

    def get_parameter_value(self, parameter_name):
        assert parameter_name == "qe_variation"
        return 0.063


@pytest.fixture
def sim_telarray_stage(corsika_config_mock_array_model):
    return SimTelArrayStage(
        simulation_parameters=corsika_config_mock_array_model.simulation_parameters,
        array_model=corsika_config_mock_array_model.array_model,
        label="test-sim-telarray-stage",
        context=corsika_config_mock_array_model.context,
    )


def _inputs(config, items):
    """Return the stage inputs of one array model for a chain."""
    return StageInputs(
        array_models=(config.array_model,),
        simulation_parameters=(config.simulation_parameters,),
        shower_configurations=(),
        args={},
        context=config.context,
        chain=parse_chain(items),
    )


def _light_inputs(light_simulation, items):
    """Return the stage inputs of a light-emission run for a chain."""
    return StageInputs(
        array_models=(),
        simulation_parameters=(),
        shower_configurations=(),
        args={},
        context=light_simulation.context,
        chain=parse_chain(items),
        calibration=light_simulation,
    )


def _set_mirror_classes(array_model, mirror_classes):
    array_model.telescope_models = {
        f"tel-{index}": _Telescope(mirror_class)
        for index, mirror_class in enumerate(mirror_classes)
    }


def test_instances_one_per_array_model(corsika_config_mock_array_model):
    config = corsika_config_mock_array_model
    inputs = StageInputs(
        array_models=(config.array_model,) * 2,
        simulation_parameters=(config.simulation_parameters,) * 2,
        shower_configurations=(),
        args={},
        context=config.context,
        chain=parse_chain(CORSIKA_OPTICS_SIM_TELARRAY),
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


def test_consumes_photons_of_its_source(sim_telarray_stage):
    assert SimTelArrayStage.consumes == (
        CHERENKOV_PHOTONS,
        CAMERA_PHOTONS_IACT,
        LIGHT_EMISSION_PHOTONS,
    )
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


def test_prepare_reads_zero_transmission_table_before_show_all(corsika_config_mock_array_model):
    config = corsika_config_mock_array_model
    stage = SimTelArrayStage(
        simulation_parameters=config.simulation_parameters,
        array_model=config.array_model,
        label="test-zero-transmission",
        context=config.context,
        bypass_optics=1,
        atmospheric_transmission="/results/atmosphere/atmosphere-zero-transmission.dat",
    )

    lines = stage.prepare(run_number=1, input_file=STDIN)

    assert (
        "-C Bypass_Optics=1 -C atmospheric_transmission="
        "/results/atmosphere/atmosphere-zero-transmission.dat -C show=all -"
    ) in lines[0]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, ALL_FACTORS),
        ({"bypass_optics": "none"}, ALL_FACTORS),
        ({"bypass_optics": "camera_entrance"}, {QUANTUM_EFFICIENCY, ATMOSPHERIC_EXTINCTION}),
        ({"bypass_optics": "camera_only"}, {QUANTUM_EFFICIENCY}),
    ],
)
def test_applied_factors_by_bypass_mode(options, expected):
    assert SimTelArrayStage.spec.applied_factors(options) == frozenset(expected)


def test_neutral_atmosphere_option_is_rejected():
    with pytest.raises(
        ValueError, match=r"Unknown option\(s\) \['neutral_atmosphere'\] of stage 'sim_telarray'"
    ):
        SimTelArrayStage.spec.applied_factors(
            {"bypass_optics": "camera_entrance", "neutral_atmosphere": True}
        )


@pytest.mark.parametrize(
    ("options", "upstream", "expected"),
    [
        (
            {"bypass_optics": "camera_entrance"},
            frozenset({ATMOSPHERIC_EXTINCTION}),
            {QUANTUM_EFFICIENCY},
        ),
        (
            {"bypass_optics": "camera_entrance"},
            frozenset(),
            {QUANTUM_EFFICIENCY, ATMOSPHERIC_EXTINCTION},
        ),
        (
            {"bypass_optics": "camera_only"},
            frozenset({ATMOSPHERIC_EXTINCTION}),
            {QUANTUM_EFFICIENCY},
        ),
    ],
)
def test_applied_factors_extinction_moves_to_upstream_stage(options, upstream, expected):
    assert SimTelArrayStage.spec.applied_factors(options, upstream=upstream) == frozenset(expected)


@pytest.mark.parametrize("value", ["bogus", 2])
def test_chain_rejects_unknown_bypass_mode(value):
    with pytest.raises(ValueError, match="bypass_optics"):
        parse_chain(["corsika", f"sim_telarray:bypass_optics={value}"])


def test_bypass_level_none_is_sim_telarray_optics():
    assert _bypass_level("none", {}) is None


def test_bypass_level_camera_only_is_level_three():
    assert _bypass_level("camera_only", {}) == 3


@pytest.mark.parametrize(("mirror_class", "level"), [(1, 1), (2, 2)])
def test_bypass_level_camera_entrance_from_mirror_class(mirror_class, level):
    telescopes = {"tel-0": _Telescope(mirror_class), "tel-1": _Telescope(mirror_class)}

    assert _bypass_level("camera_entrance", telescopes) == level


def test_bypass_level_camera_entrance_rejects_mixed_mirror_classes():
    telescopes = {"single": _Telescope(1), "dual": _Telescope(2)}

    with pytest.raises(ValueError, match="share one mirror class"):
        _bypass_level("camera_entrance", telescopes)


@pytest.mark.parametrize(
    ("items", "mirror_classes", "expected"),
    [
        (CORSIKA_OPTICS_SIM_TELARRAY, (1, 1), None),
        (CAMERA_ONLY_SHOWER, (1, 1), 3),
        (CAMERA_ENTRANCE_NO_EXTINCTION, (1, 1), 1),
        (CAMERA_ENTRANCE_NO_EXTINCTION, (2, 2), 2),
    ],
)
def test_instances_bypass_level_from_chain(
    corsika_config_mock_array_model, items, mirror_classes, expected
):
    config = corsika_config_mock_array_model
    _set_mirror_classes(config.array_model, mirror_classes)

    (stage,) = SimTelArrayStage.instances(_inputs(config, items))

    assert stage.bypass_optics == expected


def test_instances_camera_entrance_rejects_mixed_mirror_classes(corsika_config_mock_array_model):
    config = corsika_config_mock_array_model
    _set_mirror_classes(config.array_model, (1, 2))

    with pytest.raises(ValueError, match="share one mirror class"):
        SimTelArrayStage.instances(_inputs(config, CAMERA_ENTRANCE_NO_EXTINCTION))


def test_chain_without_bypass_mode_3_does_not_probe_sim_telarray(
    corsika_config_mock_array_model, mocker
):
    run = mocker.patch(
        "simtools.runtime.dependencies.subprocess.run",
        return_value=SimpleNamespace(stdout="Release: 2024.271.0 from 2024-09-27\n", stderr=""),
    )
    config = corsika_config_mock_array_model
    _set_mirror_classes(config.array_model, (2,))

    (stage,) = SimTelArrayStage.instances(_inputs(config, CORSIKA_OPTICS_SIM_TELARRAY))
    SimTelArrayStage.instances(_inputs(config, CAMERA_ENTRANCE_NO_EXTINCTION))

    run.assert_not_called()
    assert stage.bypass_optics is None


@pytest.mark.parametrize(
    ("items", "zero_table"),
    [
        (CAMERA_ENTRANCE_SHOWER, True),
        (CAMERA_ONLY_SHOWER, False),
        (CORSIKA_OPTICS_SIM_TELARRAY, False),
        (CAMERA_ENTRANCE_NO_EXTINCTION, False),
    ],
)
def test_instances_zero_transmission_follows_upstream_extinction(
    corsika_config_mock_array_model, items, zero_table
):
    config = corsika_config_mock_array_model
    _set_mirror_classes(config.array_model, (1,))

    with patch(
        "simtools.runners.sim_telarray_stage._zero_transmission", return_value="zero.dat"
    ) as zero:
        (stage,) = SimTelArrayStage.instances(_inputs(config, items))

    if zero_table:
        zero.assert_called_once_with(config.array_model.site_model, config.context)
        assert stage.atmospheric_transmission == "zero.dat"
    else:
        zero.assert_not_called()
        assert stage.atmospheric_transmission is None


@pytest.mark.parametrize(
    ("items", "min_photons"), [(CAMERA_ONLY_SHOWER, 0), (CORSIKA_OPTICS_SIM_TELARRAY, None)]
)
def test_instances_thinned_upstream_sets_min_photons_zero(
    corsika_config_mock_array_model, items, min_photons
):
    config = corsika_config_mock_array_model

    (stage,) = SimTelArrayStage.instances(_inputs(config, items))

    assert stage.simulator_array.min_photons == min_photons


@pytest.mark.parametrize(("atmosphere_option", "expected"), [("qe", True), ("none", False)])
def test_instances_efficiency_envelope_follows_atmosphere_option(
    corsika_config_mock_array_model, atmosphere_option, expected
):
    config = corsika_config_mock_array_model
    items = (
        "corsika",
        f"obdeect_atmosphere:efficiency_envelope={atmosphere_option}",
        "obdeect_optics",
        "sim_telarray:bypass_optics=camera_only",
    )

    (stage,) = SimTelArrayStage.instances(_inputs(config, items))

    assert stage.efficiency_envelope is expected


def test_instances_without_thinning_has_no_envelope(corsika_config_mock_array_model):
    config = corsika_config_mock_array_model

    (stage,) = SimTelArrayStage.instances(_inputs(config, CORSIKA_OPTICS_SIM_TELARRAY))

    assert stage.efficiency_envelope is False


def test_prepare_with_envelope_points_sim_telarray_to_scaled_tables(
    corsika_config_mock_array_model, mocker, tmp_test_directory
):
    config = corsika_config_mock_array_model
    stage = SimTelArrayStage(
        simulation_parameters=config.simulation_parameters,
        array_model=config.array_model,
        label="test-envelope",
        context=config.context,
        efficiency_envelope=True,
    )
    stage.array_model = SimpleNamespace(
        model_version="6.0.0",
        telescope_models={"LSTN-01": _EnvelopeTelescope("LSTN-01")},
    )
    mocker.patch.object(
        stage.simulator_array, "make_run_command", return_value=["sim_telarray", "command"]
    )
    stage.simulator_array.file_list = {}

    stage.prepare(run_number=2, input_file=STDIN)

    (directory,) = stage.simulator_array.include_directories
    assert directory.name == "qe-table"
    assert directory.parent.name == "6.0.0"
    assert directory.parent.parent.name == "run000002"
    (table,) = list(directory.glob("*.dat"))
    assert table.read_text(encoding="utf-8").splitlines()[0].startswith("100 ")


def test_prepare_without_envelope_sets_no_include_directories(sim_telarray_stage, mocker):
    mocker.patch.object(
        sim_telarray_stage.simulator_array, "make_run_command", return_value=["sim_telarray"]
    )
    sim_telarray_stage.simulator_array.file_list = {}

    sim_telarray_stage.prepare(run_number=1, input_file=STDIN)

    assert sim_telarray_stage.simulator_array.include_directories == ()


@pytest.mark.parametrize("tag", ["v2025-11-30-rc", "development"])
def test_mode_3_checks_simtel_build_tag(corsika_config_mock_array_model, run_context, tag):
    (run_context.software.sim_telarray_path / "build_opts.yml").write_text(
        f"simtel_tag: {tag}\n", encoding="utf-8"
    )
    inputs = replace(
        _inputs(corsika_config_mock_array_model, CAMERA_ONLY_SHOWER), context=run_context
    )
    if tag == "v2025-11-30-rc":
        with pytest.raises(ValueError, match=r"incompatible.*v2025-11-30-rc"):
            SimTelArrayStage.instances(inputs)
    else:
        (stage,) = SimTelArrayStage.instances(inputs)
        assert stage.bypass_optics == 3


def test_mode_3_warns_when_build_tag_unavailable(
    corsika_config_mock_array_model, run_context, caplog
):
    inputs = replace(
        _inputs(corsika_config_mock_array_model, CAMERA_ONLY_SHOWER), context=run_context
    )

    (stage,) = SimTelArrayStage.instances(inputs)

    assert stage.bypass_optics == 3
    assert "build tag unavailable" in caplog.text


def test_light_instances_replace_array_stages(light_simulation):
    (stage,) = SimTelArrayStage.instances(
        _light_inputs(light_simulation, ("light_emission", "sim_telarray"))
    )
    assert isinstance(stage, LightSimTelArrayStage)
    assert stage.bypass_optics is None
    assert stage.consumes == SimTelArrayStage.consumes
    assert stage.streams is False


@pytest.mark.parametrize(
    ("mirror_class", "items", "level"),
    [
        (0, ("obdeect_flasher", "sim_telarray:bypass_optics=camera_entrance"), 1),
        (2, ("obdeect_flasher", "sim_telarray:bypass_optics=camera_entrance"), 2),
        (0, ("light_emission", "sim_telarray"), None),
        (
            0,
            (
                "obdeect_illuminator",
                "obdeect_atmosphere",
                "obdeect_optics",
                "sim_telarray:bypass_optics=camera_only",
            ),
            3,
        ),
    ],
)
def test_light_instances_bypass_level_from_chain(light_simulation, mirror_class, items, level):
    light_simulation.telescope_model.get_parameter_value.return_value = mirror_class

    (stage,) = SimTelArrayStage.instances(_light_inputs(light_simulation, items))

    assert stage.bypass_optics == level


def test_light_instances_zero_transmission_with_upstream_extinction(light_simulation):
    light_simulation.telescope_model.get_parameter_value.return_value = 0
    inputs = _light_inputs(
        light_simulation,
        (
            "obdeect_illuminator",
            "obdeect_atmosphere",
            "obdeect_optics",
            "sim_telarray:bypass_optics=camera_entrance",
        ),
    )
    with patch(
        "simtools.runners.sim_telarray_stage._zero_transmission", return_value="zero.dat"
    ) as zero:
        (stage,) = SimTelArrayStage.instances(inputs)
    zero.assert_called_once_with(light_simulation.site_model, light_simulation.context)
    assert stage.atmospheric_transmission == "zero.dat"
    assert stage.bypass_optics == 1


def test_light_instances_keep_site_table_without_upstream_extinction(light_simulation):
    light_simulation.telescope_model.get_parameter_value.return_value = 0
    (stage,) = SimTelArrayStage.instances(
        _light_inputs(
            light_simulation, ("obdeect_flasher", "sim_telarray:bypass_optics=camera_entrance")
        )
    )
    assert stage.atmospheric_transmission is None


@pytest.mark.parametrize("light_source_type", ["flat_fielding", "illuminator"])
def test_light_stage_command_equals_light_emission_script(light_simulation, light_source_type):
    light_simulation.light_emission_config["light_source_type"] = light_source_type
    (stage,) = SimTelArrayStage.instances(
        _light_inputs(light_simulation, ("light_emission", "sim_telarray"))
    )
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
    stage = LightSimTelArrayStage(SimTelArrayStage, light_simulation, bypass_optics=3)
    (line,) = stage.prepare(1, input_file=Path("/in/camera.iact"))
    assert "-C input_file=/in/camera.iact" in line
    assert "-C Bypass_Optics=3" in line
    assert "rm -f" not in line


def test_light_stage_errors(light_simulation):
    stage = LightSimTelArrayStage(SimTelArrayStage, light_simulation)
    with pytest.raises(ValueError, match="reads a file and writes its file"):
        stage.prepare(1, input_file=STDIN)
    with pytest.raises(ValueError, match="reads a file and writes its file"):
        stage.prepare(1, output=StreamTo(("downstream",)))
    assert stage.environment() == {}
    assert stage.get_resources(runtime=1.5) == {"runtime": 1.5}


def test_zero_transmission_directory_differs_from_atmosphere_stage_directory():
    """Exporting the site table into the atmosphere directory would overwrite its table."""
    from simtools.runners import atmosphere_stage, sim_telarray_stage

    assert sim_telarray_stage.ZERO_TRANSMISSION_DIRECTORY != atmosphere_stage.OUTPUT_DIRECTORY
