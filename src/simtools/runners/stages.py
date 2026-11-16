"""Simulation stages and the chains built from them.

A stage is one step of a simulation, run by one program (e.g. CORSIKA, sim_telarray or an
obdeect program). Stages are connected by the data products they consume and produce, either
through files or as a stream. A chain is an ordered list of stage names. Stage classes are
imported when first used. Further stages are added with :func:`register_stage`.
``SimulationChain`` checks that each stage consumes the data product of the stage before it
when the chain is created.
"""

from dataclasses import dataclass, field
from importlib import import_module
from itertools import pairwise
from typing import Protocol

from simtools.runners import stage_types

CAMERA_EVENTS = stage_types.CAMERA_EVENTS
CHERENKOV_PHOTONS = stage_types.CHERENKOV_PHOTONS
STDIN = stage_types.STDIN
StreamTo = stage_types.StreamTo


@dataclass(frozen=True)
class StageInputs:
    """What a simulation provides to create its stage instances."""

    array_models: tuple
    simulation_parameters: tuple
    shower_configurations: tuple
    args: dict
    context: object = None
    chain_options: dict = field(default_factory=dict, hash=False, compare=False)
    stages: tuple = ()
    calibration: object = None

    def options_of(self, stage):
        """Return the options of a stage given in the chain spec (empty if there are none)."""
        return dict(self.chain_options.get(stage, {}))


class SimulationStage(Protocol):
    """One program of a simulation chain."""

    name: str
    consumes: str | tuple[str, ...] | None
    produces: str
    streams: bool

    @classmethod
    def instances(cls, inputs, label=None):
        """Return the stage instances for a run (e.g. one per array model)."""

    def prepare(self, run_number, input_file=None, output=None):
        """Write the stage inputs and return the shell lines that run the stage.

        Parameters
        ----------
        run_number : int
            Run number.
        input_file : str or Path or None
            Input file, ``STDIN`` when reading a stream, or None for the first stage.
        output : StreamTo or None
            Downstream commands when the output is streamed; None writes the stage's file.

        Returns
        -------
        list[str]
            Shell lines.
        """

    def output_file(self, run_number):
        """Return the file the stage writes when its output is not streamed."""

    def environment(self):
        """Return environment variables needed to run the stage."""

    def get_resources(self, runtime=None):
        """Return the computing resources used."""


def _lazy_stage(class_path):
    """Return a factory importing a stage class ("module:Class") when it is first used."""

    def factory():
        module, class_name = class_path.split(":")
        return getattr(import_module(module), class_name)

    return factory


STAGES = {
    "corsika": _lazy_stage("simtools.runners.corsika_stage:CorsikaStage"),
    "sim_telarray": _lazy_stage("simtools.runners.sim_telarray_stage:SimTelArrayStage"),
    "light_emission": _lazy_stage("simtools.runners.light_source_stages:LightEmissionStage"),
    "corsika_to_stream": _lazy_stage("simtools.runners.adapter_stages:CorsikaToStreamStage"),
    "stream_to_simtel": _lazy_stage("simtools.runners.adapter_stages:StreamToSimtelStage"),
    "obdeect_atmosphere": _lazy_stage("simtools.runners.atmosphere_stage:ObdeectAtmosphereStage"),
    "obdeect_optics": _lazy_stage("simtools.runners.optics_stage:ObdeectOpticsStage"),
    "obdeect_flasher": _lazy_stage("simtools.runners.light_source_stages:FlasherSourceStage"),
    "obdeect_illuminator": _lazy_stage(
        "simtools.runners.light_source_stages:IlluminatorSourceStage"
    ),
}
STAGE_SOFTWARE = {
    "corsika": frozenset({"corsika"}),
    "sim_telarray": frozenset({"sim_telarray"}),
    "light_emission": frozenset({"sim_telarray"}),
    "corsika_to_stream": frozenset({"obdeect"}),
    "stream_to_simtel": frozenset({"obdeect"}),
    "obdeect_atmosphere": frozenset({"obdeect"}),
    "obdeect_optics": frozenset({"obdeect"}),
    "obdeect_flasher": frozenset({"obdeect"}),
    "obdeect_illuminator": frozenset({"obdeect"}),
}


def register_stage(name, factory, software=()):
    """
    Register a stage under a name, replacing any stage already registered under it.

    Parameters
    ----------
    name : str
        Stage name used in simulation chains.
    factory : callable
        Function without arguments returning the stage class.
    software : iterable of str
        Software the stage needs (names of the ``simulation_software`` of the model parameters).
    """
    STAGES[name] = factory
    STAGE_SOFTWARE[name] = frozenset(software)


def stages_software(names):
    """
    Return the software needed by stages.

    Parameters
    ----------
    names : iterable of str
        Stage names.

    Returns
    -------
    frozenset of str
        Software of all stages.

    Raises
    ------
    ValueError
        If a stage is unknown.
    """
    software = set()
    for name in names:
        if name not in STAGES:
            raise ValueError(f"Unknown simulation stage {name!r}; known: {sorted(STAGES)}")
        software |= STAGE_SOFTWARE.get(name, frozenset())
    return frozenset(software)


def _as_tuple(consumed):
    """Return the products a stage consumes as a tuple."""
    if consumed is None:
        return ()
    return (consumed,) if isinstance(consumed, str) else tuple(consumed)


def get_stage_class(name) -> type[SimulationStage]:
    """Return the stage class registered under a name."""
    try:
        return STAGES[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown simulation stage {name!r}; known: {sorted(STAGES)}") from exc


@dataclass(frozen=True)
class SimulationChain:
    """Ordered simulation stages, checked for matching data products."""

    stages: tuple[str, ...]

    def __post_init__(self):
        """Check that each stage consumes what the previous stage produces."""
        if not self.stages:
            raise ValueError("A simulation chain needs at least one stage.")
        classes = [get_stage_class(name) for name in self.stages]
        for upstream, downstream in pairwise(classes):
            if upstream.produces not in _as_tuple(downstream.consumes):
                raise ValueError(
                    f"Stage {downstream.name!r} consumes {downstream.consumes!r}, "
                    f"but {upstream.name!r} produces {upstream.produces!r}."
                )

    @property
    def software(self):
        """Return the software needed by the stages of this chain."""
        return stages_software(self.stages)

    @property
    def stage_classes(self):
        """Return the stage classes in chain order."""
        return [get_stage_class(name) for name in self.stages]

    @property
    def starts_with_showers(self):
        """Return whether the first stage generates showers (consumes no input)."""
        return self.stage_classes[0].consumes is None

    @property
    def reads_input_file(self):
        """Return whether the first stage reads an input file."""
        return not self.starts_with_showers

    def has(self, name):
        """Return whether the chain contains a stage."""
        return name in self.stages
