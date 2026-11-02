"""Simulation stages and the chains built from them.

A stage is one simulation program (e.g. CORSIKA or sim_telarray). Stages are connected by the
data products they consume and produce, either through files or as a stream. A chain is an
ordered list of stage names. Stages are looked up by name in the ``STAGES`` registry, which
maps each name to a factory returning the stage class; the factories defer imports until the
stage is used. Additional stages are added with :func:`register_stage`. ``SimulationChain``
checks that each stage consumes the data product of the stage before it when the chain is
created.
"""

from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

CHERENKOV_PHOTONS = "cherenkov_photons"
CAMERA_EVENTS = "camera_events"

STDIN = "-"


@dataclass(frozen=True)
class StreamTo:
    """Output of a stage that is streamed to downstream stage commands (shell lines)."""

    commands: tuple[str, ...]


@dataclass(frozen=True)
class StageInputs:
    """What a simulation provides to create its stage instances."""

    array_models: tuple
    simulation_parameters: tuple
    shower_configurations: tuple
    args: dict
    context: object = None


class SimulationStage(Protocol):
    """One program of a simulation chain."""

    name: str
    consumes: str | None
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


def _corsika_stage():
    from simtools.runners.corsika_stage import (  # pylint: disable=import-outside-toplevel
        CorsikaStage,
    )

    return CorsikaStage


def _sim_telarray_stage():
    from simtools.runners.sim_telarray_stage import (  # pylint: disable=import-outside-toplevel
        SimTelArrayStage,
    )

    return SimTelArrayStage


STAGES = {"corsika": _corsika_stage, "sim_telarray": _sim_telarray_stage}


def register_stage(name, factory):
    """
    Register a stage under a name, replacing any stage already registered under it.

    Parameters
    ----------
    name : str
        Stage name used in simulation chains.
    factory : callable
        Function without arguments returning the stage class.
    """
    STAGES[name] = factory


_SOFTWARE_CHAINS = {
    "corsika": ("corsika",),
    "sim_telarray": ("sim_telarray",),
    "corsika_sim_telarray": ("corsika", "sim_telarray"),
}


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
            if downstream.consumes != upstream.produces:
                raise ValueError(
                    f"Stage {downstream.name!r} consumes {downstream.consumes!r}, "
                    f"but {upstream.name!r} produces {upstream.produces!r}."
                )

    @classmethod
    def from_software(cls, simulation_software):
        """Return the chain of a ``simulation_software`` value (e.g. corsika_sim_telarray)."""
        try:
            return cls(_SOFTWARE_CHAINS[simulation_software])
        except KeyError as exc:
            raise ValueError(f"Invalid simulation software: {simulation_software}") from exc

    @property
    def software(self):
        """Return the ``simulation_software`` value of this chain."""
        return "_".join(self.stages)

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
