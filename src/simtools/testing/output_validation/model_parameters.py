"""Source-neutral model-parameter output validation."""

import numpy as np

from simtools.application.model_reader import create_model_reader
from simtools.io import ascii_handler
from simtools.testing.output_validation.reference import _compare_values
from simtools.utils import general


def validate(path, rule, configuration, model_reader=None):
    """Compare a generated parameter value with its model reference.

    Parameters
    ----------
    model_reader : object, optional
        Reader selected for the current validation run. A reader is created from
        configuration only when this is omitted.
    """
    parameter_name = rule["reference_parameter_name"]
    if model_reader is None:
        model_reader = create_model_reader(configuration.get("simulation_models_path"))
    reference = model_reader.get_model_parameter(
        parameter=parameter_name,
        site=configuration.get("site"),
        array_element_name=configuration.get("telescope"),
        parameter_version=configuration.get("parameter_version"),
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
            f"Output '{path}' model parameter '{parameter_name}' differs from the model reference."
        )
