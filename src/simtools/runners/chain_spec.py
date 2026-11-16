"""Parse ordered simulation stages and options, validating connections and efficiency factors."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import yaml

from simtools.runners.stages import (
    SimulationChain,
    applied_factors_of,
    connects,
    converter_stages,
    get_stage_class,
    stages_software,
)

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
        Stage names in chain order, including the converter stages.
    options : dict
        Options of each stage, by stage name.
    """

    stages: tuple[str, ...]
    options: Mapping = field(default_factory=dict, hash=False, compare=False)

    def chain(self):
        """Return the ``SimulationChain``; raise if stages mismatch or duplicate a factor."""
        chain = SimulationChain(self.stages)
        self.check_applied_factors()
        return chain

    @property
    def software(self):
        """Return the software needed by the listed stages."""
        return stages_software(self.stages)

    def stage_options(self, stage):
        """Return the options of a stage (empty if there are none)."""
        return dict(self.options.get(stage, {}))

    def applied_factors(self):
        """
        Return the efficiency factors applied by each stage: a list of (stage, factors).

        A stage receives the factors applied by the stages before it (``upstream``), so that a
        stage applies a factor only if no upstream stage does (see ``applied_factors_of``).
        """
        applied = []
        upstream = frozenset()
        for stage in self.stages:
            factors = applied_factors_of(stage, self.stage_options(stage), upstream)
            applied.append((stage, factors))
            upstream |= factors
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

    Converter stages (role ``converter``) may be left out: between two stages whose data
    products differ, the one converter that fits is inserted. A converter is listed explicitly
    to give it options.

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
        If the list is empty, a stage is unknown or listed twice, an item is malformed, or more
        than one converter fits between two stages.
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
    for stage, stage_options in options.items():
        spec = getattr(get_stage_class(stage), "spec", None)
        if spec is not None:
            spec.checked_options(stage_options)
    return ChainSpec(_with_converters(stages), options)


def _with_converters(stages):
    """Return the stages with the converter inserted between stages whose products differ."""
    completed = [stages[0]]
    for downstream in stages[1:]:
        converter = _converter_between(completed[-1], downstream)
        if converter is not None:
            if converter in stages or converter in completed:
                raise ValueError(
                    f"Stage {converter!r} is needed between stages {completed[-1]!r} and "
                    f"{downstream!r} but is listed elsewhere in the simulation chain."
                )
            completed.append(converter)
        completed.append(downstream)
    return tuple(completed)


def _converter_between(upstream, downstream):
    """Return the converter connecting two stages, or None if none is needed or none fits."""
    upstream_class, downstream_class = get_stage_class(upstream), get_stage_class(downstream)
    if connects(upstream_class, downstream_class):
        return None
    fitting = [
        name
        for name in converter_stages()
        if connects(upstream_class, get_stage_class(name))
        and connects(get_stage_class(name), downstream_class)
    ]
    if len(fitting) > 1:
        raise ValueError(
            f"Several converters fit between stages {upstream!r} and {downstream!r}: "
            f"{fitting}; list one of them in the chain."
        )
    return fitting[0] if fitting else None


def _parse_item(item):
    """Return the stage name and the options of one chain item."""
    if not isinstance(item, str):
        raise ValueError(f"Chain item {item!r} must be a string 'STAGE[:KEY=VALUE,...]'.")
    stage, _, text = item.strip().partition(_STAGE_SEPARATOR)
    stage = stage.strip()
    if not stage:
        raise ValueError(f"Chain item {item!r} has no stage name.")
    return stage, _parse_options(item, text)


def _split_options(text):
    """Split ``KEY=VALUE,...`` at the commas outside brackets (list values keep their commas)."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        elif char == _OPTION_SEPARATOR and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def _parse_options(item, text):
    """Return the options ``KEY=VALUE,...`` of a chain item."""
    options = {}
    for option in filter(None, (part.strip() for part in _split_options(text))):
        key, separator, value = option.partition(_VALUE_SEPARATOR)
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"Option {option!r} of chain item {item!r} must be KEY=VALUE.")
        if key in options:
            raise ValueError(f"Option {key!r} is given twice in chain item {item!r}.")
        options[key] = yaml.safe_load(value)
    return options
