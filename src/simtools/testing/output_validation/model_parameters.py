"""Database-backed model-parameter output validation."""

import numpy as np

from simtools.db import db_handler
from simtools.io import ascii_handler
from simtools.testing.output_validation.reference import _compare_values
from simtools.utils import general


def validate(path, rule, configuration):
    """Compare a generated parameter value with its database reference."""
    parameter_name = rule["reference_parameter_name"]
    reference = db_handler.DatabaseHandler().get_model_parameter(
        parameter=parameter_name,
        site=configuration.get("site"),
        array_element_name=configuration.get("telescope"),
        model_version=configuration.get("model_version"),
    )
    generated_value = ascii_handler.collect_data_from_file(path)["value"]
    reference_value = reference[parameter_name]["value"]
    scaling = rule.get("scaling", 1.0)
    comparable_value = (
        general.convert_string_to_list(generated_value)
        if isinstance(generated_value, str)
        else generated_value
    )
    generated_value = comparable_value
    if not np.isclose(scaling, 1.0):
        try:
            generated_value = np.asarray(comparable_value) * scaling
        except (TypeError, ValueError) as exc:
            raise AssertionError(
                f"Output '{path}' model parameter '{parameter_name}' cannot be scaled by {scaling}."
            ) from exc
    if not _compare_values(generated_value, reference_value, rule["tolerance"], True):
        raise AssertionError(
            f"Output '{path}' model parameter '{parameter_name}' differs from the database."
        )
