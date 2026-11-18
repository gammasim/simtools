"""Catalog of the simulation software and the stages of a simulation chain.

``resources/simulation_software.yml`` is the only place that names the simulation programs, the
stages of a chain, the data products that connect them, the stage options and the files the
programs write (the files are merged into the file catalog, see ``simtools.io.file_type``). Code
asks this catalog for a stage's role instead of naming a stage or a program.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from importlib.resources import files

import jsonschema
import yaml

from simtools.io.file_type import SIMULATION_SOFTWARE_CATALOG, product_value

_SCHEMA = "schemas/simulation_software.schema.yml"
#: Kinds of photons a first stage generates.
AIR_SHOWER_SOURCE = "air_shower"
LIGHT_SOURCE = "light_source"
#: Data product of the ray-tracing stages; they select the ray-tracing backend, not run in a chain.
STAR_IMAGES = "star_images"
#: Python types of the option types.
_OPTION_TYPES = {
    "integer": (int,),
    "number": (int, float),
    "string": (str,),
    "boolean": (bool,),
    "list": (list, tuple),
}


def _read(path):
    return yaml.safe_load(files("simtools").joinpath(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def catalog():
    """Return the validated catalog of the simulation software and stages."""
    data = _read(SIMULATION_SOFTWARE_CATALOG)
    jsonschema.Draft7Validator(_read(_SCHEMA)).validate(data)
    _check_references(data)
    return data


def _check(names, known, what, where):
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise ValueError(f"{where}: unknown {what} {unknown}; known: {sorted(known)}")


def _check_stage(name, stage, data):
    where = f"Stage '{name}' of the software catalog"
    _check([stage["software"]], data["software"], "software", where)
    _check([*stage.get("consumes", ()), stage["produces"]], data["products"], "products", where)
    factors = [*stage.get("applies", ()), *stage.get("skips_if_applied_upstream", ())]
    for values in stage.get("applies_by_option", {}).values():
        for value in values.values():
            applies, skips = _value_factors(value)
            factors.extend([*applies, *skips])
    _check(factors, data["efficiency_factors"], "efficiency factors", where)
    options = stage.get("options", {})
    _check(
        [*stage.get("applies_by_option", {}), *stage.get("camera_only", {})],
        options,
        "options",
        where,
    )
    if "light_source" in stage:
        _check([stage["light_source"]], data.get("light_sources", {}), "light sources", where)
    for product_id in [*stage.get("files", {}).values(), *[stage.get("production_result")]]:
        if product_id is not None:
            product_value(product_id, "role")


def _value_factors(value):
    """Return (applies, skips_if_applied_upstream) of an ``applies_by_option`` value."""
    if isinstance(value, dict):
        return tuple(value["applies"]), tuple(value.get("skips_if_applied_upstream", ()))
    return tuple(value), ()


def _check_references(data):
    """Check that the names a catalog entry uses are declared in the catalog."""
    for name, stage in data["stages"].items():
        _check_stage(name, stage, data)
    for kind, chain in data["default_chains"].items():
        _check(chain, data["stages"], "stages", f"Default chain '{kind}'")


@dataclass(frozen=True)
class StageSpec:
    """
    A stage of a simulation chain as declared in the software catalog.

    Parameters
    ----------
    name : str
        Stage name used in simulation chains.
    entry : dict
        Catalog entry of the stage.
    """

    name: str
    entry: dict = field(repr=False, hash=False, compare=False)

    def __getattr__(self, key):
        """Return a field of the catalog entry (None if the entry does not declare it)."""
        if key.startswith("_"):
            raise AttributeError(key)
        return self.entry.get(key)

    @property
    def software(self):
        """Software the stage runs."""
        return frozenset({self.entry["software"]})

    @property
    def consumes(self):
        """Products the stage reads (None for a first stage)."""
        consumed = self.entry.get("consumes")
        return tuple(consumed) if consumed else None

    @property
    def options(self):
        """Options of the stage by name."""
        return self.entry.get("options", {})

    def file_type(self, use):
        """Return the file product of a use of the stage (``output``, ``log``, ...)."""
        try:
            return self.entry["files"][use]
        except KeyError as exc:
            raise ValueError(f"Stage {self.name!r} declares no {use} file.") from exc

    def checked_options(self, options):
        """
        Return the stage options with their defaults, after checking names, types and values.

        Parameters
        ----------
        options : dict
            Options given in the chain.

        Raises
        ------
        ValueError
            For an unknown option, a wrong type or a value not allowed.
        """
        if self.options_from_arguments:
            return dict(options)
        unknown = sorted(set(options) - set(self.options))
        if unknown:
            raise ValueError(
                f"Unknown option(s) {unknown} of stage {self.name!r}; known: {sorted(self.options)}"
            )
        checked = {
            key: option["default"] for key, option in self.options.items() if "default" in option
        }
        for key, value in options.items():
            option = self.options[key]
            if option["type"] == "list" and not isinstance(value, (list, tuple)):
                value = [value]  # one value of a list option
            _check_option(self.name, key, option, value)
            checked[key] = value
        return checked

    def applied_factors(self, options, upstream=frozenset()):
        """
        Return the efficiency factors the stage applies (catalog ``applies``).

        ``applies_by_option`` selects the factors by an option value; factors listed in
        ``skips_if_applied_upstream`` are left out when an upstream stage applies them.
        """
        factors, _skippable = self._declared_factors(options)
        return factors - self.skipped_factors(options, upstream)

    def skipped_factors(self, options, upstream=frozenset()):
        """Return the factors the stage would apply but leaves to an upstream stage."""
        factors, skippable = self._declared_factors(options)
        return frozenset(factors & skippable & set(upstream))

    def _declared_factors(self, options):
        """Return the factors the options select, and those an upstream stage may apply instead."""
        options = self.checked_options(options)
        factors = set(self.entry.get("applies", ()))
        skippable = set(self.entry.get("skips_if_applied_upstream", ()))
        for option, by_value in self.entry.get("applies_by_option", {}).items():
            applies, skips = _value_factors(by_value.get(options.get(option), ()))
            factors |= set(applies)
            skippable |= set(skips)
        return frozenset(factors), frozenset(skippable)

    def simulates_camera_only(self, options):
        """Return True if the options make the stage simulate the camera only."""
        camera_only = self.entry.get("camera_only")
        if not camera_only:
            return False
        options = self.checked_options(options)
        return all(options.get(key) == value for key, value in camera_only.items())


def _check_option(stage, key, option, value):
    """Raise ValueError if an option value does not match its declaration."""
    expected = _OPTION_TYPES[option["type"]]
    wrong_type = not isinstance(value, expected) or (
        option["type"] in ("integer", "number") and isinstance(value, bool)
    )
    if not wrong_type and option["type"] == "list":
        items = _OPTION_TYPES.get(option.get("items"), (object,))
        wrong_type = not all(isinstance(item, items) for item in value) or (
            "length" in option and len(value) != option["length"]
        )
    if wrong_type:
        raise ValueError(
            f"Option {key!r} of stage {stage!r} must be of type {option['type']}"
            + (f" of {option['items']}" if "items" in option else "")
            + (f" with {option['length']} values" if "length" in option else "")
            + f"; got {value!r}."
        )
    if "enum" in option and value not in option["enum"]:
        raise ValueError(
            f"Invalid {key} option of stage {stage!r}: {value!r}; choose from {option['enum']}."
        )


def stage_names():
    """Return the names of the stages of the catalog."""
    return tuple(catalog()["stages"])


def stage_spec(name):
    """Return the catalog declaration of a stage; raise ValueError if it is unknown."""
    try:
        return StageSpec(name, catalog()["stages"][name])
    except KeyError as exc:
        raise ValueError(
            f"Unknown simulation stage {name!r}; known: {sorted(stage_names())}"
        ) from exc


def software(name):
    """Return the catalog entry of a simulation program."""
    try:
        return catalog()["software"][name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown simulation software {name!r}; known: {sorted(catalog()['software'])}"
        ) from exc


def software_names():
    """Return the names of the simulation programs."""
    return tuple(catalog()["software"])


def product(name):
    """Return the catalog entry of a data product."""
    return catalog()["products"][name]


def default_chain(source):
    """Return the default chain of a kind of source (``air_shower`` or ``light_source``)."""
    return tuple(catalog()["default_chains"][source])


def light_source(name):
    """Return the catalog entry of a light-source type."""
    sources = catalog().get("light_sources", {})
    try:
        return sources[name]
    except KeyError as exc:
        raise ValueError(f"Unknown light source type {name!r}; known: {sorted(sources)}") from exc


def light_source_names():
    """Return the names of the light-source types."""
    return tuple(catalog().get("light_sources", {}))


def is_star_stage(name):
    """Return True if a catalog stage produces star images, i.e. selects a ray-tracing backend."""
    return name in stage_names() and stage_spec(name).produces == STAR_IMAGES


def star_backend(items=None):
    """
    Return the ray-tracing backend selected by the star stages of a list of stages.

    Parameters
    ----------
    items : str, list of str or None
        Stage items as in ``simulation_software`` (e.g. ``["sim_telarray_star"]``).

    Returns
    -------
    str or None
        Software of the star stage; None if the list has no star stage.

    Raises
    ------
    ValueError
        If the list has more than one star stage.
    """
    if not items:
        return None
    # stage names only: parse_chain would look up stage classes, and star stages have none
    items = [items] if isinstance(items, str) else list(items)
    names = [item.split(":")[0] for item in items]  # STAGE or STAGE:KEY=VALUE,...
    star = [stage_spec(name) for name in names if is_star_stage(name)]
    if len(star) > 1:
        raise ValueError(f"Only one ray-tracing stage is allowed, got {[s.name for s in star]}.")
    return next(iter(star[0].software)) if star else None


def load_object(path):
    """Return the Python object of a catalog reference ``module:attribute``."""
    module, attribute = path.split(":")
    return getattr(import_module(module), attribute)
