"""Chain specifications: which stage programs make up a simulation, selected by name or file.

A chain spec is an ordered list of stage names (see ``simtools.runners.stages``) with options
for each stage. Each stage consumes and produces a named data product; adapters between data
products are ordinary stages. A spec is either a preset or a YAML file::

    stages:
      - stage: corsika
      - stage: <adapter or atmosphere stage>
        options: {...}
      - stage: sim_telarray

The preset ``legacy`` is CORSIKA piped into sim_telarray, with sim_telarray's own atmosphere,
optics and light emission. It is the default and is built from no other package. Stages of other
simulation programs register themselves (``stages.register_stage``) and presets
(``register_preset``); the spec only refers to them by name.

A stage class may define ``applied_factors(options)``, returning the efficiency factors it
applies to the photons (e.g. ``"mirror_reflectivity"``). A chain in which two stages apply the
same factor is rejected, so that no factor is applied twice.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from simtools.io import ascii_handler
from simtools.runners.stages import SimulationChain, get_stage_class

LEGACY_PRESET = "legacy"

#: Efficiency factors that can be applied by more than one program of a chain.
ATMOSPHERIC_EXTINCTION = "atmospheric_extinction"
MIRROR_REFLECTIVITY = "mirror_reflectivity"
TELESCOPE_TRANSMISSION = "telescope_transmission"
QUANTUM_EFFICIENCY = "quantum_efficiency"

_PRESETS = {}


@dataclass(frozen=True)
class ChainSpec:
    """
    Stages of a simulation chain and their options.

    Parameters
    ----------
    name : str
        Preset name, or the file name of a spec file.
    stages : tuple of str
        Stage names in chain order.
    options : dict
        Options of each stage, by stage name.
    """

    name: str
    stages: tuple[str, ...]
    options: Mapping = field(default_factory=dict, hash=False, compare=False)

    def chain(self):
        """Return the ``SimulationChain``; raise if two stages apply the same factor."""
        chain = SimulationChain(self.stages)
        self.check_applied_factors()
        return chain

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
                        f"Chain {self.name!r} applies {factor!r} twice: "
                        f"in stage {seen[factor]!r} and in stage {stage!r}."
                    )
                seen[factor] = stage


def register_preset(name, stages, options=None):
    """
    Register a chain preset, replacing any preset of the same name.

    Parameters
    ----------
    name : str
        Preset name (value of the chain selection).
    stages : tuple of str
        Stage names in chain order.
    options : dict, optional
        Options of each stage, by stage name.
    """
    _PRESETS[name] = ChainSpec(name, tuple(stages), dict(options or {}))


def preset_names():
    """Return the names of the registered presets."""
    return tuple(_PRESETS)


def load_chain_spec(selection=None):
    """
    Return the chain spec of a preset name or of a YAML file.

    Parameters
    ----------
    selection : str or Path, optional
        Preset name or spec file; the default is the legacy preset.

    Returns
    -------
    ChainSpec
        Chain specification.

    Raises
    ------
    ValueError
        If the selection is neither a preset nor an existing file, or the file is malformed.
    """
    selection = LEGACY_PRESET if selection is None else selection
    if str(selection) in _PRESETS:
        return _PRESETS[str(selection)]
    path = Path(selection)
    if not path.is_file():
        raise ValueError(
            f"Unknown simulation chain {str(selection)!r}: not a preset "
            f"({', '.join(preset_names())}) and not a file."
        )
    return _spec_from_file(path)


def _spec_from_file(path):
    """Read a chain spec file."""
    entries = ascii_handler.collect_data_from_file(path).get("stages")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Chain spec {path} needs a non-empty 'stages' list.")
    stages, options = [], {}
    for entry in entries:
        name = entry.get("stage") if isinstance(entry, dict) else None
        if not isinstance(name, str):
            raise ValueError(f"Chain spec {path}: each entry needs a 'stage' name, got {entry!r}.")
        stages.append(name)
        if entry.get("options"):
            options[name] = dict(entry["options"])
    return ChainSpec(str(path), tuple(stages), options)


register_preset(LEGACY_PRESET, ("corsika", "sim_telarray"))
