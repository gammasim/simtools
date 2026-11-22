"""Parse and validate simulation chains: ordered stages, their options and efficiency factors."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise

import yaml

from simtools.runners.stages import (
    AIR_SHOWER_SOURCE,
    applied_factors_of,
    connects,
    converter_stages,
    get_stage_class,
    stage_role,
    stages_software,
    stages_with,
)
from simtools.runtime import software_catalog

_STAGE_SEPARATOR = ":"
_OPTION_SEPARATOR = ","
_VALUE_SEPARATOR = "="


@dataclass(frozen=True)
class SimulationChain:
    """
    Ordered simulation stages and their options, checked when the chain is created.

    The chain is valid if each stage consumes what the stage before it produces (with its
    options), no efficiency factor is applied twice, and a chain ending in camera events applies
    every efficiency factor required for its photon source.

    Parameters
    ----------
    stages : tuple of str
        Stage names in chain order, including the converter stages.
    options : dict
        Checked options of each stage, with their defaults, by stage name.
    """

    stages: tuple[str, ...]
    options: Mapping = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self):
        """Check the connections and the efficiency factors of the chain."""
        if not self.stages:
            raise ValueError("A simulation chain needs at least one stage.")
        for upstream, downstream in pairwise(self.stages):
            self._check_connection(upstream, downstream)
        self._check_applied_factors()
        self._check_required_factors()

    def _check_connection(self, upstream, downstream):
        """Raise ValueError if a stage does not consume the product of the stage before it."""
        upstream_class, downstream_class = get_stage_class(upstream), get_stage_class(downstream)
        if not connects(upstream_class, downstream_class):
            raise ValueError(
                f"Stage {downstream!r} consumes {downstream_class.consumes!r}, "
                f"but {upstream!r} produces {upstream_class.produces!r}."
            )
        spec = getattr(downstream_class, "spec", None)
        if spec is None:
            return
        consumed = spec.consumes_with(self.stage_options(downstream))
        if upstream_class.produces not in consumed:
            raise ValueError(
                f"Stage {downstream!r} with options {self.stage_options(downstream)} consumes "
                f"{list(consumed)}, but {upstream!r} produces {upstream_class.produces!r}; "
                f"set the options of {downstream!r} for this product."
            )

    def _check_applied_factors(self):
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

    def _check_required_factors(self):
        """Raise ValueError if a chain ending in camera events misses an efficiency factor."""
        produces = get_stage_class(self.stages[-1]).produces
        try:
            format_ = software_catalog.product(produces)["format"]
        except KeyError:
            return
        if format_ != software_catalog.EVENTS_FORMAT:
            return
        required = software_catalog.required_efficiency_factors(self.source or AIR_SHOWER_SOURCE)
        missing = sorted(required - self.upstream_factors(None))
        if missing:
            raise ValueError(
                f"The chain {list(self.stages)} does not apply the efficiency factors {missing}."
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
    def source(self):
        """Return the kind of photons the first stage generates (None if it reads an input)."""
        return stage_role(self.stage_classes[0], "source")

    @property
    def generates_air_showers(self):
        """Return whether the first stage generates air showers."""
        return self.source == AIR_SHOWER_SOURCE

    @property
    def reads_input_file(self):
        """Return whether the first stage reads an input file (it generates no photons)."""
        return self.stage_classes[0].consumes is not None

    def runs(self, implementation):
        """Return whether a stage of the chain is run by an implementation (a stage class)."""
        return any(issubclass(stage, implementation) for stage in self.stage_classes)

    def with_role(self, role, value=True):
        """Return the stages of the chain that have a role (see ``stages.stages_with``)."""
        return stages_with(self.stages, role, value)

    def stage_options(self, stage):
        """Return the checked options of a stage, with their defaults (empty if it has none)."""
        return dict(self.options.get(stage, {}))

    def applied_factors(self):
        """
        Return the efficiency factors applied by each stage: a list of (stage, factors).

        A stage receives the factors applied by the stages before it, so that a stage leaves a
        factor to an upstream stage where its catalog entry allows it.
        """
        applied = []
        upstream = frozenset()
        for stage in self.stages:
            factors = applied_factors_of(stage, self.stage_options(stage), upstream)
            applied.append((stage, factors))
            upstream |= factors
        return applied

    def upstream_factors(self, stage):
        """
        Return the efficiency factors applied by the stages before a stage.

        Parameters
        ----------
        stage : str or None
            Stage of the chain; None returns the factors applied by all stages.
        """
        upstream = frozenset()
        for name, factors in self.applied_factors():
            if name == stage:
                break
            upstream |= factors
        return upstream

    def skipped_factors(self, stage):
        """Return the efficiency factors a stage leaves to an upstream stage."""
        spec = getattr(get_stage_class(stage), "spec", None)
        if spec is None:
            return frozenset()
        return spec.skipped_factors(self.stage_options(stage), self.upstream_factors(stage))

    def upstream_stages(self, stage):
        """Return the stages before a stage of the chain."""
        return self.stages[: self.stages.index(stage)]


def parse_chain(items):
    """
    Return the simulation chain of a list of stage items.

    Converter stages (role ``converter``) may be left out: between two stages whose data
    products differ, the one converter that fits is inserted. A converter is listed explicitly
    to give it options.

    Parameters
    ----------
    items : str or list of str
        Items ``STAGE`` or ``STAGE:KEY=VALUE[,KEY=VALUE...]`` in chain order.

    Returns
    -------
    SimulationChain
        Checked simulation chain.

    Raises
    ------
    ValueError
        If the list is empty, a stage is unknown or listed twice, an item or an option is
        malformed, more than one converter fits between two stages, or the chain is invalid.
    """
    items = [items] if isinstance(items, str) else list(items or ())
    if not items:
        raise ValueError("A simulation chain needs at least one stage.")
    stages, given = [], {}
    for item in items:
        stage, stage_options = _parse_item(item)
        if stage in stages:
            raise ValueError(f"Stage {stage!r} is listed twice in the simulation chain.")
        stages.append(stage)
        given[stage] = stage_options
    stages = _with_converters(stages)
    return SimulationChain(
        stages, {stage: _checked(stage, given.get(stage, {})) for stage in stages}
    )


def _checked(stage, options):
    """Return the options of a stage checked against its catalog entry, with their defaults."""
    spec = getattr(get_stage_class(stage), "spec", None)
    return dict(options) if spec is None else spec.checked_options(options)


def _with_converters(stages):
    """Return the stages with the converter inserted between stages whose products differ."""
    completed = list(stages[:1])
    for upstream, downstream in pairwise(stages):
        converter = _converter_between(upstream, downstream)
        if converter is not None:
            if converter in stages or converter in completed:
                raise ValueError(
                    f"Stage {converter!r} is needed between stages {upstream!r} and "
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
