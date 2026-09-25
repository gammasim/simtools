"""Validation helpers for JSON-compatible structured model values."""

import math
from numbers import Real


def validate_finite_json_values(value, path="$"):
    """Reject non-finite JSON numbers and identify their nested path.

    Parameters
    ----------
    value : object
        JSON-like value to inspect.
    path : str, optional
        JSON path used in an error message.

    Raises
    ------
    ValueError
        If a real number is NaN or infinite.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            validate_finite_json_values(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_finite_json_values(item, f"{path}[{index}]")
    elif isinstance(value, Real) and not isinstance(value, bool) and not math.isfinite(value):
        raise ValueError(f"Non-finite JSON number at {path}")
