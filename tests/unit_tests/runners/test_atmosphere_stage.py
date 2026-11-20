"""Tests for the obdeect atmosphere simulation stage."""

import shlex
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import pytest
from astropy.table import Column, Table

from simtools.runners import stages
from simtools.runners.atmosphere_stage import run_seed
from simtools.runners.chain_spec import parse_chain
from simtools.runners.stages import STDIN, StageInputs, StreamTo, get_stage_class
from simtools.simulation_software.atmosphere import AtmosphereRequest
from simtools.simulation_software.programs import program_command

PHOTONS_GROUND = "photons_ground"
PHOTONS_SKY = "photons_sky"
ATMOSPHERIC_EXTINCTION = "atmospheric_extinction"
QUANTUM_EFFICIENCY = "quantum_efficiency"
CAMERA_ONLY_CHAIN = ["corsika", "obdeect_atmosphere", "obdeect_optics"]
ObdeectAtmosphereStage = get_stage_class("obdeect_atmosphere")


class _RunFiles:
    """File naming of a run: ``<directory>/run<N>.<stage><suffix>``."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def get_stage_file_name(self, stage_name, file_type, run_number=None):
        assert file_type == "photon_stream"
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / f"run{run_number:06d}.{stage_name}.pst"


@pytest.fixture(autouse=True)
def pass_transmission_through(monkeypatch):
    """Instances tests use model tables that are not on disk: keep the table path."""
    monkeypatch.setattr(
        "simtools.runners.atmosphere_stage.write_obdeect_transmission",
        lambda transmission_file, _directory: transmission_file,
    )


@pytest.fixture
def request_for_stage(tmp_test_directory):
    return AtmosphereRequest(
        site="North",
        observation_level_m=2158.0,
        profile_file=Path(tmp_test_directory) / "atmprof.dat",
        transmission_file=Path(tmp_test_directory) / "transmission.dat",
        seed=3,
        telescope_pathlen_m={1: 28.0},
    )


@pytest.fixture
def atmosphere_stage(request_for_stage, run_context):
    return ObdeectAtmosphereStage(
        request_for_stage,
        run_files=_RunFiles(run_context.paths.get_output_directory("corsika")),
        context=run_context,
    )


def _tokens(line):
    return shlex.split(line)


def _inputs(run_context, array_model, chain_items=("obdeect_atmosphere",), **kwargs):
    """Stage inputs of one array model and a chain given by its items."""
    return StageInputs(
        array_models=(array_model,),
        simulation_parameters=(),
        shower_configurations=(),
        args=kwargs.pop("args", {}),
        context=run_context,
        chain=parse_chain(list(chain_items)),
        **kwargs,
    )


def _instance(inputs, run_context):
    """Return the atmosphere stage of the inputs, with the run files of a corsika run."""
    (stage,) = ObdeectAtmosphereStage.instances(inputs)
    if stage.run_files is None:
        stage.run_files = _RunFiles(run_context.paths.get_output_directory("corsika"))
    return stage


def test_stage_protocol_attributes():
    assert ObdeectAtmosphereStage.name == "obdeect_atmosphere"
    assert ObdeectAtmosphereStage.consumes == (PHOTONS_SKY, PHOTONS_GROUND)
    assert ObdeectAtmosphereStage.produces == PHOTONS_GROUND
    assert ObdeectAtmosphereStage.streams is True


def test_applied_factors_is_atmospheric_extinction():
    assert ObdeectAtmosphereStage.spec.applied_factors({}) == frozenset({ATMOSPHERIC_EXTINCTION})
    assert ObdeectAtmosphereStage.spec.applied_factors({"seed": 2}) == frozenset(
        {ATMOSPHERIC_EXTINCTION}
    )


def test_prepare_reads_stdin_and_writes_file(atmosphere_stage, run_context):
    lines = atmosphere_stage.prepare(run_number=4, input_file=STDIN)

    output = atmosphere_stage.output_file(4)
    assert len(lines) == 1
    assert _tokens(lines[0]) == [
        *program_command("obdeect", "obdeect-propagation-run"),
        "--transmission",
        str(atmosphere_stage.request.transmission_file),
        "--site-altitude",
        "2158.0",
        "--profile",
        str(atmosphere_stage.request.profile_file),
        "--seed",
        "3",
        "--mode",
        "thinned",
        "--telescope-pathlen",
        "1=28.0",
        "-i",
        "-",
        "-o",
        str(output),
    ]
    assert output.parent.is_dir()
    assert output.parent == run_context.paths.get_output_directory("corsika")


def test_prepare_reads_input_file(atmosphere_stage):
    lines = atmosphere_stage.prepare(run_number=1, input_file="corsika_photons.pst")

    tokens = _tokens(lines[0])
    assert tokens[tokens.index("-i") + 1] == "corsika_photons.pst"


def test_prepare_pipes_streamed_output_into_downstream_command(atmosphere_stage):
    downstream = "sim_telarray -C input_file=-"

    lines = atmosphere_stage.prepare(run_number=1, input_file=STDIN, output=StreamTo((downstream,)))

    assert len(lines) == 1
    assert lines[0].endswith(f" | {downstream}")
    tokens = _tokens(lines[0].split(" | ")[0])
    assert tokens[tokens.index("-o") + 1] == "-"


def test_prepare_rejects_several_downstream_commands(atmosphere_stage):
    with pytest.raises(ValueError, match=r"pipes into one downstream command, got 2"):
        atmosphere_stage.prepare(
            run_number=1, input_file=STDIN, output=StreamTo(("cmd_a", "cmd_b"))
        )


def test_prepare_without_input_raises(atmosphere_stage):
    with pytest.raises(ValueError, match=r"needs an input photon stream"):
        atmosphere_stage.prepare(run_number=1, input_file=None)


def test_output_file_name_uses_stage_and_run(atmosphere_stage):
    assert atmosphere_stage.output_file(7).name == "run000007.obdeect_atmosphere.pst"


def test_output_file_needs_run_files(request_for_stage, run_context):
    stage = ObdeectAtmosphereStage(request_for_stage, context=run_context)

    with pytest.raises(ValueError, match="no run configuration to name its output file"):
        stage.output_file(1)


def test_instances_create_one_stage_for_the_array_model(array_model_north, run_context):
    inputs = _inputs(run_context, array_model_north)

    stage_list = ObdeectAtmosphereStage.instances(inputs, label="test")

    assert len(stage_list) == 1
    (stage,) = stage_list
    assert stage.request.site == "North"
    assert stage.request.seed == 1
    assert stage.seed_per_run
    assert stage.request.mode == "thinned"


def test_run_seed_depends_on_base_seed_and_run_number():
    assert run_seed(22, 1) == run_seed(22, 1)
    assert len({run_seed(22, 1), run_seed(22, 2), run_seed(23, 1)}) == 3


def test_instances_derive_the_seed_of_each_run(array_model_north, run_context, monkeypatch):
    monkeypatch.setattr(
        array_model_north, "sim_telarray_seed", SimpleNamespace(seed_string="11,22")
    )
    stage = _instance(_inputs(run_context, array_model_north), run_context)

    assert stage.request.seed == 22
    for run_number in (1, 2):
        tokens = _tokens(stage.prepare(run_number=run_number, input_file=STDIN)[0])
        assert tokens[tokens.index("--seed") + 1] == str(run_seed(22, run_number))


def test_instances_use_stage_options(array_model_north, run_context):
    chain_items = ["obdeect_atmosphere:seed=7,mode=weighted,wavelength_range_nm=[250.0,700.0]"]
    stage = _instance(_inputs(run_context, array_model_north, chain_items), run_context)

    assert stage.request.seed == 7
    assert not stage.seed_per_run
    for run_number in (1, 2):
        tokens = _tokens(stage.prepare(run_number=run_number, input_file=STDIN)[0])
        assert tokens[tokens.index("--seed") + 1] == "7"
    assert stage.request.mode == "weighted"
    assert stage.request.wavelength_range_nm == (250.0, 700.0)


def test_chain_options_reject_unknown_option():
    with pytest.raises(
        ValueError, match=r"Unknown option\(s\) \['threads'\] of stage 'obdeect_atmosphere'"
    ):
        parse_chain(["obdeect_atmosphere:threads=4"])


def test_chain_options_reject_invalid_mode():
    with pytest.raises(ValueError, match=r"Invalid mode option of stage 'obdeect_atmosphere'"):
        parse_chain(["obdeect_atmosphere:mode=sparse"])


class _CalibrationModel:
    """Calibration device model with one monochromatic wavelength (flasher_wavelength)."""

    def get_parameter_value_with_unit(self, name):
        assert name == "flasher_wavelength"
        return [300.0 * u.nm]


def _calibration_inputs(run_context, array_model, chain_items, args=None):
    return _inputs(
        run_context,
        array_model,
        chain_items,
        args={} if args is None else args,
        calibration=SimpleNamespace(
            calibration_model=_CalibrationModel(),
            runner_service=_RunFiles(run_context.paths.get_output_directory("light")),
        ),
    )


def test_instances_illuminator_range_is_source_wavelength(array_model_north, run_context):
    inputs = _calibration_inputs(run_context, array_model_north, ["obdeect_atmosphere"])

    stage = _instance(inputs, run_context)

    assert stage.request.wavelength_range_nm == pytest.approx((300.0, 300.0))


def test_instances_illuminator_rejects_unsupported_wavelength(array_model_north, run_context):
    inputs = _calibration_inputs(
        run_context, array_model_north, ["obdeect_atmosphere"], args={"wavelength": [400.0]}
    )

    with pytest.raises(ValueError, match=r"is not supported by the light source"):
        ObdeectAtmosphereStage.instances(inputs)


def test_instances_explicit_range_overrides_source_wavelength(array_model_north, run_context):
    inputs = _calibration_inputs(
        run_context,
        array_model_north,
        ["obdeect_atmosphere:wavelength_range_nm=[250.0,700.0]"],
    )

    stage = _instance(inputs, run_context)

    assert stage.request.wavelength_range_nm == (250.0, 700.0)


def test_instances_without_light_source_keeps_run_keys(array_model_north, run_context):
    stage = _instance(_inputs(run_context, array_model_north), run_context)

    assert stage.request.wavelength_range_nm is None


def test_chain_accepts_photons_from_upstream_stage(fake_stage_factories):
    fake_stage_factories("fake_sky_source", None, PHOTONS_SKY)

    chain = parse_chain(["fake_sky_source", "obdeect_atmosphere"])

    assert chain.stage_classes[-1] is ObdeectAtmosphereStage


def test_chain_rejects_upstream_product_it_does_not_consume(fake_stage_factories):
    fake_stage_factories("fake_camera", None, "camera_events")

    with pytest.raises(ValueError, match=r"'obdeect_atmosphere' consumes"):
        parse_chain(["fake_camera", "obdeect_atmosphere"])


@pytest.fixture
def fake_stage_factories(monkeypatch):
    """Register fake upstream stages (stages of other programs) by name."""

    def _register(name, consumes, produces):
        fake = type(
            f"Fake_{name}",
            (),
            {"name": name, "consumes": consumes, "produces": produces, "streams": True},
        )
        monkeypatch.setitem(stages.registry(), name, fake)

    return _register


def test_bypass_camera_entrance_with_upstream_extinction_applies_extinction_once():
    chain = parse_chain([*CAMERA_ONLY_CHAIN, "sim_telarray:bypass_optics=camera_entrance"])

    applied = dict(chain.applied_factors())
    assert applied["obdeect_atmosphere"] == frozenset({ATMOSPHERIC_EXTINCTION})
    assert applied["sim_telarray"] == frozenset({QUANTUM_EFFICIENCY})


def test_bypass_camera_entrance_without_upstream_atmosphere_applies_extinction_in_sim_telarray():
    chain = parse_chain(["corsika", "obdeect_optics", "sim_telarray:bypass_optics=camera_entrance"])

    applied = dict(chain.applied_factors())
    assert applied["sim_telarray"] == frozenset({QUANTUM_EFFICIENCY, ATMOSPHERIC_EXTINCTION})


class _EnvelopeTelescope:
    """Telescope model with a QE table and a QE variation (what the envelope reads)."""

    def __init__(self, name, wavelength, efficiency):
        self.name = name
        self.parameters = {"quantum_efficiency": {"value": "qe.ecsv", "instrument": name}}
        self._table = Table({"wavelength": Column(wavelength, unit="nm"), "efficiency": efficiency})

    def get_parameter_table(self, parameter_name):
        assert parameter_name == "quantum_efficiency"
        return self._table

    def get_parameter_value(self, parameter_name):
        assert parameter_name == "qe_variation"
        return 0.063


@pytest.fixture
def envelope_model():
    telescope = _EnvelopeTelescope("LSTN-01", [300.0, 500.0], [0.2, 0.4])
    return SimpleNamespace(model_version="6.0.0", telescope_models={"LSTN-01": telescope})


def test_prepare_with_envelope_writes_bound_and_passes_it(
    request_for_stage, run_context, envelope_model
):
    stage = ObdeectAtmosphereStage(
        request_for_stage,
        run_files=_RunFiles(run_context.paths.get_output_directory("corsika")),
        context=run_context,
        envelope_model=envelope_model,
    )

    lines = stage.prepare(run_number=2, input_file=STDIN)

    tokens = _tokens(lines[0])
    envelope = Path(tokens[tokens.index("--envelope") + 1].removeprefix("qe="))
    assert envelope.name == "qe-envelope.dat"
    assert envelope.parent == run_context.paths.get_output_directory(
        ["efficiency", "run000002", "6.0.0"]
    )
    # two nodes per nm bin on the 100-2000 nm grid
    assert len(envelope.read_text(encoding="utf-8").splitlines()) == 2 * 1901


def test_prepare_without_envelope_has_no_envelope_option(atmosphere_stage):
    lines = atmosphere_stage.prepare(run_number=1, input_file=STDIN)

    assert "--envelope" not in _tokens(lines[0])


def test_envelope_off_by_default_for_chain_not_ending_camera_only(array_model_north, run_context):
    stage = _instance(_inputs(run_context, array_model_north, ["obdeect_atmosphere"]), run_context)

    assert stage.envelope_model is None


def test_envelope_on_by_default_for_camera_only_chain(array_model_north, run_context):
    chain_items = [*CAMERA_ONLY_CHAIN, "sim_telarray:bypass_optics=camera_only"]

    stage = _instance(_inputs(run_context, array_model_north, chain_items), run_context)

    assert stage.envelope_model is array_model_north


def test_envelope_on_by_option(array_model_north, run_context):
    chain_items = ["obdeect_atmosphere:efficiency_envelope=qe"]

    stage = _instance(_inputs(run_context, array_model_north, chain_items), run_context)

    assert stage.envelope_model is array_model_north


def test_envelope_off_by_option_for_camera_only_chain(array_model_north, run_context):
    chain_items = [
        "corsika",
        "obdeect_atmosphere:efficiency_envelope=none",
        "obdeect_optics",
        "sim_telarray:bypass_optics=camera_only",
    ]

    stage = _instance(_inputs(run_context, array_model_north, chain_items), run_context)

    assert stage.envelope_model is None


def test_envelope_option_rejects_invalid_value():
    with pytest.raises(ValueError, match=r"Invalid efficiency_envelope option of stage"):
        parse_chain(["obdeect_atmosphere:efficiency_envelope=maybe"])


@pytest.mark.parametrize(
    "atmosphere_item", ["obdeect_atmosphere", "obdeect_atmosphere:efficiency_envelope=none"]
)
def test_calibration_does_not_thin_by_qe_envelope(run_context, atmosphere_item):
    chain_items = [
        "obdeect_illuminator",
        atmosphere_item,
        "obdeect_optics",
        "sim_telarray:bypass_optics=camera_only",
    ]
    inputs = StageInputs(
        array_models=(),
        simulation_parameters=(),
        shower_configurations=(),
        args={},
        context=run_context,
        calibration=object(),
        chain=parse_chain(chain_items),
    )

    assert not ObdeectAtmosphereStage.efficiency_envelope(inputs)


def test_calibration_rejects_explicit_qe_envelope(run_context):
    inputs = StageInputs(
        array_models=(),
        simulation_parameters=(),
        shower_configurations=(),
        args={},
        context=run_context,
        calibration=object(),
        chain=parse_chain(["obdeect_atmosphere:efficiency_envelope=qe"]),
    )

    with pytest.raises(ValueError, match=r"efficiency envelope.*calibration"):
        ObdeectAtmosphereStage.efficiency_envelope(inputs)
