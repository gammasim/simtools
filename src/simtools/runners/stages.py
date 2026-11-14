"""Simulation stages and the chains built from them.

A stage is one step of a simulation, run by one program (e.g. CORSIKA, sim_telarray or an
obdeect program). Stages are connected by the data products they consume and produce, either
through files or as a stream. A chain is an ordered list of stage names. The stages are declared
in ``resources/simulation_stages.yml`` (stage class and the software it needs); the classes are
imported when a stage is first used. Further stages are added with :func:`register_stage`.
``SimulationChain`` checks that each stage consumes the data product of the stage before it
when the chain is created.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from importlib.resources import files
from itertools import pairwise
from typing import Protocol

from simtools.io import ascii_handler
from simtools.simulation_chain import products

CHERENKOV_PHOTONS = products.CHERENKOV_PHOTONS
CAMERA_EVENTS = products.CAMERA_EVENTS
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


STAGE_CATALOG = "resources/simulation_stages.yml"


def _lazy_stage(class_path):
    """Return a factory importing a stage class ("module:Class") when it is first used."""

    def factory():
        module, class_name = class_path.split(":")
        return getattr(import_module(module), class_name)

    return factory


@lru_cache(maxsize=1)
def stage_catalog():
    """Return the stage declarations of the packaged catalog, by stage name."""
    return ascii_handler.collect_data_from_file(files("simtools").joinpath(STAGE_CATALOG))["stages"]


STAGES = {name: _lazy_stage(entry["class"]) for name, entry in stage_catalog().items()}
STAGE_SOFTWARE = {
    name: frozenset(entry.get("software", ())) for name, entry in stage_catalog().items()
}
STAGE_CONVERTERS = {name for name, entry in stage_catalog().items() if entry.get("converter")}


def register_stage(name, factory, software=(), converter=False):
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
    converter : bool
        The stage only converts one data product into another; it is inserted between stages
        whose products do not match (see :func:`insert_converters`).
    """
    STAGES[name] = factory
    STAGE_SOFTWARE[name] = frozenset(software)
    STAGE_CONVERTERS.discard(name)
    if converter:
        STAGE_CONVERTERS.add(name)


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


def _converter_paths(produced, accepted):
    """
    Return the shortest converter sequences from a product to one of the accepted products.

    A sequence is a tuple of converter names. The search is breadth first over the products:
    a converter leads from each product it consumes to the product it produces.
    """
    accepted = set(accepted)
    paths = {produced: [()]}
    frontier = [produced]
    while frontier:
        found = [path for product in frontier if product in accepted for path in paths[product]]
        if found:
            return found
        reached = {}
        for product in frontier:
            for name in sorted(STAGE_CONVERTERS):
                converter = get_stage_class(name)
                if product in _as_tuple(converter.consumes) and converter.produces not in paths:
                    reached.setdefault(converter.produces, []).extend(
                        (*path, name) for path in paths[product]
                    )
        paths.update(reached)
        frontier = list(reached)
    return []


def insert_converters(names):
    """
    Return the stage names with the converters inserted where two stages do not match.

    Where a stage produces a product that the next stage does not consume, the shortest
    sequence of converter stages from the one product to the other is inserted. Stages that match
    are left as they are, and so are converters listed explicitly. If there is no such sequence
    the chain is left as it is, and the chain validation reports the mismatch.

    Parameters
    ----------
    names : iterable of str
        Stage names in chain order.

    Returns
    -------
    tuple of str
        Stage names with the converters inserted.

    Raises
    ------
    ValueError
        If two different converter sequences of the same length connect two stages.
    """
    names = tuple(names)
    resolved = list(names[:1])
    for downstream in names[1:]:
        upstream_class = get_stage_class(resolved[-1])
        accepted = _as_tuple(get_stage_class(downstream).consumes)
        if upstream_class.produces not in accepted:
            paths = _converter_paths(upstream_class.produces, accepted)
            if len(paths) > 1:
                options = "; ".join(" -> ".join(path) for path in paths)
                raise ValueError(
                    f"Several converters connect {resolved[-1]!r} and {downstream!r}: {options}. "
                    "List the converter that should be used in the chain."
                )
            resolved.extend(paths[0] if paths else ())
        resolved.append(downstream)
    return tuple(resolved)


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
