import pytest

from simtools.simulation.nsb import airmass_scale


@pytest.mark.parametrize(
    ("zenith", "expected"),
    [(0.0, 1.0), (20.0, 1.0330008543), (60.0, 1.3500071125)],
)
def test_airmass_scale(zenith, expected):
    assert airmass_scale(zenith, 0.84, 0.29) == pytest.approx(expected, rel=1e-6)
