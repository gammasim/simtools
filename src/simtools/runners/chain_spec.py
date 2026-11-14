"""Chain specification: the stages of a simulation and their options, given as a list.

A chain is an ordered list of stage names (see ``simtools.runners.stages``), each with optional
options, written ``STAGE`` or ``STAGE:KEY=VALUE[,KEY=VALUE...]``. The same list is used on the
command line (``--simulation_chain corsika sim_telarray``) and in configuration files::

    simulation_chain:
      - corsika
      - corsika_to_stream
      - obdeect_atmosphere
      - obdeect_optics
      - stream_to_simtel
      - sim_telarray:bypass_optics=3

Option values are read as YAML scalars (``3`` is an integer, ``true`` a boolean). A stage is
listed at most once.

A stage class may define ``applied_factors(options)``, returning the efficiency factors it
applies to the photons (e.g. ``"mirror_reflectivity"``). A chain in which two stages apply the
same factor is rejected, so that no factor is applied twice.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import yaml

from simtools.runners.stages import SimulationChain, get_stage_class, stages_software

#: Efficiency factors that can be applied by more than one program of a chain.
ATMOSPHERIC_EXTINCTION = "atmospheric_extinction"
MIRROR_REFLECTIVITY = "mirror_reflectivity"
TELESCOPE_TRANSMISSION = "telescope_transmission"
QUANTUM_EFFICIENCY = "quantum_efficiency"

_STAGE_SEPARATOR = ":"
_OPTION_SEPARATOR = ","
_VALUE_SEPARATOR = "="


@dataclass(frozen=True)
class ChainSpec:
    """
    Stages of a simulation chain and their options.

    Parameters
    ----------
    stages : tuple of str
        Stage names in chain order.
    options : dict
        Options of each stage, by stage name.
    """

    stages: tuple[str, ...]
    options: Mapping = field(default_factory=dict, hash=False, compare=False)

    def chain(self):
        """Return the ``SimulationChain``; raise if two stages apply the same factor."""
        chain = SimulationChain(self.stages)
        self.check_applied_factors()
        return chain

    @property
    def software(self):
        """Return the software needed by the stages (without loading the stage classes)."""
        return stages_software(self.stages)

    def stage_options(self, stage):
        """Return the options of a stage (empty if there are none)."""
        return dict(self.options.get(stage, {}))

    def applied_factors(self):
        """Return the efficiency factors applied by each stage: a list of (stage, factors)."""
        applied = []
        for stage in self.stages:
            method = getattr(get_stage_class(stage), "applied_factors", None)
            factors = frozenset(method(self.stage_options(stage))) if method else frozenset()
            applied.append((stage, factors))
        return applied

    def check_applied_factors(self):
        """Raise ValueError if two stages apply the same efficiency factor."""
        seen = {}
        for stage, factors in self.applied_factors():
            for factor in sorted(factors):
                if factor in seen:
                    raise ValueError(
                        f"The chain applies {factor!r} twice: "
                        f"in stage {seen[factor]!r} and in stage {stage!r}."
                    )
                seen[factor] = stage


def parse_chain(items):
    """
    Return the chain specification of a list of stage items.

    Parameters
    ----------
    items : str or list of str
        Items ``STAGE`` or ``STAGE:KEY=VALUE[,KEY=VALUE...]`` in chain order.

    Returns
    -------
    ChainSpec
        Chain specification.

    Raises
    ------
    ValueError
        If the list is empty, a stage is listed twice, or an item is malformed.
    """
    items = [items] if isinstance(items, str) else list(items or ())
    if not items:
        raise ValueError("A simulation chain needs at least one stage.")
    stages, options = [], {}
    for item in items:
        stage, stage_options = _parse_item(item)
        if stage in stages:
            raise ValueError(f"Stage {stage!r} is listed twice in the simulation chain.")
        stages.append(stage)
        if stage_options:
            options[stage] = stage_options
    return ChainSpec(tuple(stages), options)


def _parse_item(item):
    """Return the stage name and the options of one chain item."""
    if not isinstance(item, str):
        raise ValueError(f"Chain item {item!r} must be a string 'STAGE[:KEY=VALUE,...]'.")
    stage, _, text = item.strip().partition(_STAGE_SEPARATOR)
    stage = stage.strip()
    if not stage:
        raise ValueError(f"Chain item {item!r} has no stage name.")
    return stage, _parse_options(item, text)


def _parse_options(item, text):
    """Return the options ``KEY=VALUE,...`` of a chain item."""
    options = {}
    for option in filter(None, (part.strip() for part in text.split(_OPTION_SEPARATOR))):
        key, separator, value = option.partition(_VALUE_SEPARATOR)
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"Option {option!r} of chain item {item!r} must be KEY=VALUE.")
        if key in options:
            raise ValueError(f"Option {key!r} is given twice in chain item {item!r}.")
        options[key] = yaml.safe_load(value)
    return options
