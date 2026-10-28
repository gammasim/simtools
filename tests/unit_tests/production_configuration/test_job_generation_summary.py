import astropy.units as u

from simtools.production_configuration.job_generation_summary import (
    format_integer_summary,
    format_quantity_summary,
)


def test_format_quantity_summary():
    assert format_quantity_summary([5 * u.deg, 5 * u.deg]) == "5 deg"
    assert format_quantity_summary([1 * u.deg, 3 * u.deg]) == "[1, 3] deg"


def test_format_integer_summary():
    assert format_integer_summary([3, 3]) == "3"
    assert format_integer_summary([2, 4]) == "[2, 4]"
