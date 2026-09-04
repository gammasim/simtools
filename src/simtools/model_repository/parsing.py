"""Parsing helpers shared by simulation-model sources."""

from simtools.utils import value_conversion


def normalize_model_parameter(data):
    """Normalize the value and unit fields of a model-parameter document."""
    data = dict(data)
    data["value"], _ = value_conversion.split_value_and_unit(
        data["value"], "int" in data.get("type", "float")
    )
    data["value"], base_unit, _ = value_conversion.get_value_unit_type(
        value=data["value"], unit_str=data.get("unit")
    )
    data["unit"] = value_conversion.normalize_model_parameter_unit(data["value"], base_unit)
    return data
