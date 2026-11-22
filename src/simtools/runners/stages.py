"""Simulation stages and the chains built from them.

A stage is one step of a simulation, run by one program (e.g. CORSIKA, sim_telarray or an
obdeect program). Stages are connected by the data products they consume and produce, either
through files or as a stream. A chain is an ordered list of stage names.

The stages are declared in ``resources/simulation_software.yml`` (see
``simtools.runtime.software_catalog``): name, software, products, options, files and role. The
class of a stage is its catalog ``implementation``; it holds only how the stage runs. Other code
asks a stage for its role (:func:`stages_with`, :func:`stage_role`) instead of naming it.
``simtools.runners.chain_spec.SimulationChain`` checks that each stage consumes the data product
of the stage before it when the chain is created. Converter stages only change the format of
photons; a chain may leave them out.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from simtools.job_execution.runner_services import RunnerServices
from simtools.runners import stage_types
from simtools.runtime import software_catalog

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
    chain: Any = None  # SimulationChain (simtools.runners.chain_spec)
    calibration: object = None

    @property
    def stages(self):
        """Return the stage names of the chain (empty without a chain)."""
        return () if self.chain is None else self.chain.stages

    def options_of(self, stage):
        """Return the checked options of a stage, with their defaults (empty without a chain)."""
        return {} if self.chain is None else self.chain.stage_options(stage)


class SimulationStage(Protocol):
    """
    One program of a simulation chain.

    The name, the products and the catalog entry (``spec``, see
    ``simtools.runtime.software_catalog.StageSpec``) come from the software catalog. The roles a
    stage can have in a chain are catalog fields (:func:`stage_role`): ``software``, ``source``
    (``air_shower`` or ``light_source``), ``converter``, ``thins_photons``,
    ``single_array_model``, ``required_arguments``, ``production_result`` and the efficiency
    factors it applies.
    """

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


class CatalogStage:
    """
    Base class of the stage implementations of the software catalog.

    The registry makes a subclass per catalog stage that sets the name, the products and the
    catalog entry (``spec``); an implementation holds only how the stage runs.
    """

    name = None
    consumes = None
    produces = None
    spec = None
    streams = True

    def environment(self):
        """Return environment variables needed to run the stage (none)."""
        return {}

    def get_resources(self):
        """Return computing resources used (none: the first stage of a chain reports them)."""
        return {}


AIR_SHOWER_SOURCE = software_catalog.AIR_SHOWER_SOURCE
LIGHT_SOURCE = software_catalog.LIGHT_SOURCE
#: Stage classes by name: the catalog stages (built on first use, see :func:`registry`) and the
#: stages added with :func:`register_stage`.
STAGES = {}


def _catalog_stage_class(spec):
    """
    Return the class of a catalog stage: a subclass of its implementation with the catalog fields.

    The subclass carries the name, the products and the catalog entry (``spec``), so that one
    implementation can serve several stages (e.g. the light-source stages).
    """
    implementation = software_catalog.load_object(spec.implementation)
    return type(
        implementation.__name__,
        (implementation,),
        {
            "name": spec.name,
            "consumes": spec.consumes,
            "produces": spec.produces,
            "spec": spec,
            "__module__": implementation.__module__,
            "__doc__": implementation.__doc__,
        },
    )


@lru_cache(maxsize=1)
def _catalog_stages():
    """Return the classes of the catalog stages by name (imported once)."""
    return {
        name: _catalog_stage_class(software_catalog.stage_spec(name))
        for name in software_catalog.stage_names()
    }


def registry():
    """Return the stage classes by name: the catalog stages and the registered stages."""
    for name, stage_class in _catalog_stages().items():
        STAGES.setdefault(name, stage_class)
    return STAGES


def register_stage(stage_class):
    """
    Register a stage class under its name, replacing any stage registered under it.

    A stage of simtools is an entry of ``resources/simulation_software.yml``; this adds a stage
    outside the catalog (e.g. in tests). The class declares ``name``, ``consumes`` and
    ``produces``, and the role attributes it has (see :class:`SimulationStage`).

    Parameters
    ----------
    stage_class : type
        Stage class.
    """
    registry()[stage_class.name] = stage_class


def stage_role(stage_class, role, default=None):
    """
    Return a role of a stage class: its catalog field, or else its attribute.

    Parameters
    ----------
    stage_class : type
        Stage class.
    role : str
        Role (see :class:`SimulationStage`).
    default : object
        Value if the stage does not declare the role.
    """
    spec = getattr(stage_class, "spec", None)
    value = getattr(spec, role) if spec is not None else getattr(stage_class, role, None)
    return default if value is None else value


def stages_with(names, role, value=True):
    """
    Return the names of the stages that have a role.

    Parameters
    ----------
    names : iterable of str
        Stage names (e.g. the stages of a chain).
    role : str
        Role attribute (see :class:`SimulationStage`).
    value : object
        Value of the role attribute.

    Returns
    -------
    list of str
        Names of the stages whose role attribute equals ``value``, in the given order.
    """
    return [name for name in names if stage_role(get_stage_class(name), role) == value]


def converter_stages():
    """Return the names of the converter stages (role ``converter``)."""
    return [name for name, stage in registry().items() if stage_role(stage, "converter", False)]


def applied_factors_of(stage, options, upstream=frozenset()):
    """
    Return the efficiency factors a stage applies, given the factors applied upstream.

    The factors follow the catalog declaration of the stage (``applies``, ``applies_by_option``,
    ``skips_if_applied_upstream``); a stage outside the catalog applies none.
    """
    spec = getattr(get_stage_class(stage), "spec", None)
    return spec.applied_factors(options, upstream) if spec is not None else frozenset()


def files_of_run(inputs, label=None, required=True):
    """
    Return the file naming of a run (``RunnerServices``), shared by the stages of its chain.

    A light-emission run names its files after the light-emission simulation; a shower run after
    the run of its first stage (run type: the software of that stage; the first shower
    configuration).

    Parameters
    ----------
    inputs : StageInputs
        Inputs of the stages.
    label : str, optional
        Instance label.
    required : bool
        Raise if the run has no configuration; otherwise return None (a stage that only streams
        writes no file).
    """
    if inputs.calibration is not None:
        return inputs.calibration.runner_service
    if not inputs.shower_configurations or not inputs.stages:
        if not required:
            return None
        raise ValueError("The stages of a chain need a light-emission or a shower configuration.")
    config = inputs.shower_configurations[0]
    (run_type,) = stage_role(get_stage_class(inputs.stages[0]), "software")
    return RunnerServices(
        config, run_type=run_type, label=label, context=inputs.context or config.context
    )


def stage_file(stage, use, run_number):
    """
    Return a file of a stage instance: ``<run base name>.<stage name><suffix>``.

    The file product of the use (``output``, ``log``, ...) is the catalog ``files`` entry of the
    stage; the instance has the file naming of its run (``run_files``).

    Parameters
    ----------
    stage : object
        Stage instance with ``spec`` and ``run_files``.
    use : str
        Use of the file.
    run_number : int
        Run number.
    """
    if stage.run_files is None:
        raise ValueError(f"Stage {stage.name!r} has no run configuration to name its {use} file.")
    return stage.run_files.get_stage_file_name(
        stage.name, stage.spec.file_type(use), run_number=run_number
    )


def single_downstream(output, stage_name):
    """Return the one downstream command of a streamed output; raise if there are several."""
    if len(output.commands) != 1:
        raise ValueError(
            f"Stage {stage_name!r} pipes into one downstream command, got {len(output.commands)}."
        )
    return output.commands[0]


def ends_camera_only(inputs):
    """
    Return True if the last stage of the chain simulates the camera only.

    The last stage declares this in the catalog (``camera_only``: option values).

    Parameters
    ----------
    inputs : StageInputs
        Inputs of the stages (chain stages and chain options).
    """
    if not inputs.stages:
        return False
    last = inputs.stages[-1]
    spec = getattr(get_stage_class(last), "spec", None)
    return bool(spec and spec.simulates_camera_only(inputs.options_of(last)))


def check_array_models(stages, array_models):
    """
    Raise ValueError if a stage that runs for one array model gets several.

    Parameters
    ----------
    stages : iterable of str
        Stage names of the chain.
    array_models : sequence
        Array models of the run (one per model version).
    """
    single = stages_with(stages, "single_array_model")
    if single and len(array_models) > 1:
        raise ValueError(
            f"Stages {single} simulate one model version per run; "
            f"got {len(array_models)} model versions."
        )


def efficiency_envelope_enabled(inputs):
    """
    Return True if a thinning stage of the chain thins the photons by the efficiency envelope.

    A thinning stage declares this with the role method ``efficiency_envelope(inputs)``.

    Parameters
    ----------
    inputs : StageInputs
        Inputs of the stages (chain stages and chain options).
    """
    for name in stages_with(inputs.stages, "thins_photons"):
        envelope = getattr(get_stage_class(name), "efficiency_envelope", None)
        if envelope and envelope(inputs):
            return True
    return False


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
        software |= frozenset(stage_role(get_stage_class(name), "software", ()))
    return frozenset(software)


def _as_tuple(consumed):
    """Return the products a stage consumes as a tuple."""
    if consumed is None:
        return ()
    return (consumed,) if isinstance(consumed, str) else tuple(consumed)


def connects(upstream, downstream):
    """Return whether stage class ``downstream`` consumes what stage class ``upstream`` produces."""
    return upstream.produces in _as_tuple(downstream.consumes)


def get_stage_class(name) -> type[SimulationStage]:
    """Return the stage class registered under a name."""
    stages = registry()
    try:
        return stages[name]
    except KeyError as exc:
        raise ValueError(f"Unknown simulation stage {name!r}; known: {sorted(stages)}") from exc
