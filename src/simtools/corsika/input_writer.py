"""Serialize model parameter values for CORSIKA input cards."""

from functools import cache

import yaml
from astropy import units as u

from simtools.constants import SCHEMA_PATH


def _first_interaction_height(value, unit):
    """Format the CORSIKA FIXHEI value."""
    return [f"{value * u.Unit(unit).to('cm'):.2f}", "0"]


def _starting_grammage(value, unit):
    """Format the CORSIKA FIXCHI value."""
    return f"{value * u.Unit(unit).to('g/cm2')}"


def _energy_cutoffs(value, unit):
    """Format the four CORSIKA ECUTS values."""
    converted = [item * u.Unit(unit).to("GeV") for item in value]
    return [" ".join(str(item) for item in converted)]


def _longitudinal_step(value, unit):
    """Format the CORSIKA LONGI value."""
    return ["T", str(value * u.Unit(unit).to("g/cm2")), "F", "F"]


def _wavelength_range(value, unit):
    """Format the two CORSIKA CWAVLG values."""
    return [str(item * u.Unit(unit).to("nm")) for item in value]


def _io_buffer(value, unit):
    """Format the CORSIKA IO_BUFFER value."""
    megabytes = value * u.Unit(unit).to("Mbyte")
    if megabytes.is_integer():
        return f"{int(megabytes)}MB"
    return str(int(value * u.Unit(unit).to("byte")))


CONVERTERS = {
    "first_interaction_height": _first_interaction_height,
    "starting_grammage": _starting_grammage,
    "energy_cutoffs": _energy_cutoffs,
    "longitudinal_step": _longitudinal_step,
    "wavelength_range": _wavelength_range,
    "io_buffer": _io_buffer,
}


@cache
def _schema_formats(parameter_name):
    """Return declared CORSIKA formats from a model-parameter schema."""
    schema_path = SCHEMA_PATH / "model_parameters" / f"{parameter_name}.schema.yml"
    with schema_path.open(encoding="utf-8") as schema_file:
        documents = yaml.safe_load_all(schema_file)
        schemas = [document for document in documents if document]
    formats = {}
    for schema in schemas:
        for software in schema.get("simulation_software", []):
            if software.get("name") == "corsika" and software.get("format"):
                formats[schema["name"]] = software["format"]
    return formats


def format_parameter(parameter_name, value, unit=None):
    """Format a value using the CORSIKA format declared in its parameter schema."""
    try:
        format_name = _schema_formats(parameter_name)[parameter_name]
    except (KeyError, FileNotFoundError) as exc:
        raise ValueError(f"No CORSIKA input format is declared for {parameter_name!r}.") from exc
    return format_value(value, format_name, unit)


def format_value(value, format_name, unit=None):
    """Format a value using a schema-declared CORSIKA input format."""
    if format_name == "scalar":
        return [value]
    if format_name == "list":
        return list(value)
    if format_name == "bool_flag":
        return ["T" if value else "F"]
    if format_name.startswith("quantity:"):
        target_unit = format_name.split(":", 1)[1]
        quantity = value * u.Unit(unit)
        return (
            [str(item) for item in quantity.to(target_unit).value]
            if hasattr(value, "__iter__")
            else [str(quantity.to_value(target_unit))]
        )
    if format_name.startswith("converter:"):
        converter_name = format_name.split(":", 1)[1]
        try:
            converter = CONVERTERS[converter_name]
        except KeyError as exc:
            raise ValueError(f"Unknown CORSIKA input converter: {converter_name}") from exc
        return converter(value, unit)
    raise ValueError(f"Unknown CORSIKA input format: {format_name}")
