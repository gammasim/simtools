import pytest

from simtools.production_configuration.angle_ranges import (
    ceil_with_tolerance,
    directed_circular_span_degrees,
)


def test_directed_circular_span_degrees_returns_directed_span_for_non_wrapping_range():
    assert directed_circular_span_degrees((10, 110)) == pytest.approx(100.0)


def test_directed_circular_span_degrees_returns_directed_span_for_wrapping_range():
    assert directed_circular_span_degrees((350, 10)) == pytest.approx(20.0)


def test_directed_circular_span_degrees_treats_full_circle_as_360_degrees():
    assert directed_circular_span_degrees((0, 360)) == pytest.approx(360.0)
    assert directed_circular_span_degrees((15, 375)) == pytest.approx(360.0)


def test_ceil_with_tolerance_rounds_near_integer_without_overshooting():
    assert ceil_with_tolerance(10.00000000001) == 10


def test_ceil_with_tolerance_uses_ceiling_for_non_integer_values():
    assert ceil_with_tolerance(10.2) == 11
