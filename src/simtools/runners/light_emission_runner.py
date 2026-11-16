"""Build calibration run scripts from explicitly ordered simulation stages."""

from dataclasses import dataclass
from types import SimpleNamespace

from simtools.runners import chain_spec
from simtools.runners.chain_runner import ChainRunner
from simtools.runners.stages import (
    AIR_SHOWER_SOURCE,
    LIGHT_SOURCE,
    StageInputs,
    get_stage_class,
    registry,
    stage_role,
)


@dataclass(frozen=True)
class LightArray:
    """Models of the one telescope of a light-emission simulation (as an array model)."""

    telescope_models: dict
    site_model: object
    model_version: str
    site: str
    sim_telarray_seed: object


@dataclass(frozen=True)
class LightPointing:
    """Pointing of the telescope of a light-emission simulation (degrees)."""

    zenith_angle: float
    azimuth_angle: float


def check_light_chain(chain):
    """
    Check that a chain is a chain of a calibration light source.

    Parameters
    ----------
    chain : SimulationChain
        Chain to check.

    Raises
    ------
    ValueError
        If the chain does not start with a light-source stage or has an air-shower stage.
    """
    if chain.source != LIGHT_SOURCE or chain.with_role("source", AIR_SHOWER_SOURCE):
        first = chain.stages[0]
        light = sorted(
            name
            for name, stage in registry().items()
            if stage_role(stage, "source") == LIGHT_SOURCE
        )
        raise ValueError(
            f"A chain of a calibration light source starts with a light-source stage ({light}) "
            f"and has no air-shower stage; got {list(chain.stages)} (first stage {first!r} "
            f"consumes {get_stage_class(first).consumes!r})."
        )


def _seed_object(seeds):
    """Return the seeds (instrument, simulation) as the seed object of an array model."""
    instrument, simulation = seeds
    return SimpleNamespace(
        seed_string=f"{instrument},{simulation}",
        instrument_seed=instrument,
        simulation_seed=simulation,
    )


def stage_inputs(calibration, spec, chain):
    """
    Return the stage inputs of a light-emission simulation.

    Parameters
    ----------
    calibration : SimulatorLightEmission
        Light-emission simulation of one telescope and one light source.
    spec : ChainSpec
        Chain specification (stage options).
    chain : SimulationChain
        Simulation chain.

    Returns
    -------
    StageInputs
        Inputs of the stages; the telescope of the simulation is the only array element.
    """
    config = dict(calibration.light_emission_config)
    config["telescope"] = calibration.telescope
    config["light_source"] = config.get("light_source") or calibration.calibration_model.name
    array = LightArray(
        telescope_models={calibration.telescope: calibration.telescope_model},
        site_model=calibration.site_model,
        model_version=calibration.telescope_model.model_version,
        site=calibration.site_model.site,
        sim_telarray_seed=_seed_object(calibration.simulation_seeds()),
    )
    theta, phi = calibration.telescope_pointing()
    return StageInputs(
        array_models=(array,),
        simulation_parameters=(LightPointing(float(theta), float(phi)),),
        shower_configurations=(),
        args=config,
        context=calibration.context,
        chain_options=spec.options,
        stages=chain.stages,
        calibration=calibration,
    )


def build_chain_runner(calibration, items, label=None):
    """
    Return the chain runner of a light-emission simulation.

    Parameters
    ----------
    calibration : SimulatorLightEmission
        Light-emission simulation of one telescope and one light source.
    items : list of str
        Stages of the chain, each ``STAGE`` or ``STAGE:KEY=VALUE,...``.
    label : str, optional
        Label of the stages.

    Returns
    -------
    ChainRunner
        Chain runner.
    """
    spec = chain_spec.parse_chain(items)
    chain = spec.chain()
    check_light_chain(chain)
    return ChainRunner(
        chain, stage_inputs(calibration, spec, chain), label=label or calibration.label
    )


def write_chain_script(calibration, items, script_file):
    """
    Write the run script of the stages of a chain for a light-emission simulation.

    Parameters
    ----------
    calibration : SimulatorLightEmission
        Light-emission simulation of one telescope and one light source.
    items : list of str
        Stages of the chain, each ``STAGE`` or ``STAGE:KEY=VALUE,...``.
    script_file : str or Path
        Script to write.

    Returns
    -------
    ChainRunner
        Chain runner (file lists of the stages).
    """
    runner = build_chain_runner(calibration, items)
    runner.prepare_run(
        run_number=calibration.light_emission_config.get("run_number") or 1,
        sub_script=script_file,
    )
    return runner
