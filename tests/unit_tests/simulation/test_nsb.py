import pytest

from simtools.simulation.nsb import (
    NsbSettings,
    airmass_scale,
    nsb_rate_factor,
    scale_nsb_parameters,
    site_nsb_factor,
)


@pytest.mark.parametrize(
    ("zenith", "expected"),
    [(0.0, 1.0), (20.0, 1.0330008543), (60.0, 1.3500071125)],
)
def test_airmass_scale(zenith, expected):
    assert airmass_scale(zenith, 0.84, 0.29) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        (NsbSettings(zenith_angle=20.0, manual_factor=None, autoscale=True), 0.20660017),
        (NsbSettings(zenith_angle=20.0, manual_factor=2.0, autoscale=True), 0.4),
        (NsbSettings(zenith_angle=20.0, manual_factor=None, autoscale=False), 0.2),
    ],
)
def test_scale_nsb_parameters_preserves_reference(settings, expected):
    parameters = {
        "nsb_pixel_rate": {"value": [0.2, 0.4]},
        "nsb_autoscale_airmass": {"value": [0.84, 0.29]},
    }
    result = scale_nsb_parameters(parameters, settings, 1.0)
    assert result["nsb_pixel_rate"]["value"] == pytest.approx([expected, expected * 2])
    assert result["nsb_autoscale_airmass"]["value"] == [0.0, 0.29]
    assert parameters["nsb_pixel_rate"]["value"] == [0.2, 0.4]
    assert parameters["nsb_autoscale_airmass"]["value"] == [0.84, 0.29]


@pytest.mark.parametrize(
    ("rate", "expected_rate"),
    [
        ("0.2", 0.20660017),
        ("0.2 0.4", [0.20660017, 0.41320034]),
        ("0.2, 0.4", [0.20660017, 0.41320034]),
    ],
)
def test_scale_nsb_parameters_from_string_values(rate, expected_rate):
    parameters = {
        "nsb_pixel_rate": {"value": rate},
        "nsb_autoscale_airmass": {"value": "0.84 0.29"},
    }
    settings = NsbSettings(zenith_angle=20.0, manual_factor=None, autoscale=True)

    result = scale_nsb_parameters(parameters, settings, 1.0)

    assert result["nsb_pixel_rate"]["value"] == pytest.approx(expected_rate)
    assert result["nsb_autoscale_airmass"]["value"] == [0.0, 0.29]
    assert parameters["nsb_pixel_rate"]["value"] == rate
    assert parameters["nsb_autoscale_airmass"]["value"] == "0.84 0.29"


@pytest.mark.parametrize(
    ("site_parameters", "expected"),
    [({}, 1.0), ({"nsb_scaling_factor": {"value": "3.0"}}, 3.0)],
)
def test_site_nsb_factor(site_parameters, expected):
    assert site_nsb_factor(site_parameters) == pytest.approx(expected)


def test_scale_nsb_parameters_uses_site_factor_without_manual_factor():
    parameters = {
        "nsb_pixel_rate": {"value": 0.2},
        "nsb_autoscale_airmass": {"value": [0.0, 0.29]},
        "nsb_scaling_factor": {"value": 2.0},
    }
    settings = NsbSettings(zenith_angle=20.0, manual_factor=None, autoscale=True)

    result = scale_nsb_parameters(parameters, settings, 2.0)

    assert result["nsb_pixel_rate"]["value"] == pytest.approx(0.4)
    assert result["nsb_scaling_factor"]["value"] == pytest.approx(1.0)
    assert parameters["nsb_scaling_factor"]["value"] == pytest.approx(2.0)


@pytest.mark.parametrize("airmass", ["0.84", [0.84, 0.29, 0.1]])
def test_scale_nsb_parameters_autoscale_airmass_requires_two_values(airmass):
    parameters = {
        "nsb_pixel_rate": {"value": 0.2},
        "nsb_autoscale_airmass": {"value": airmass},
    }
    settings = NsbSettings(zenith_angle=20.0, manual_factor=None, autoscale=True)

    with pytest.raises(ValueError, match="nsb_autoscale_airmass requires two values"):
        scale_nsb_parameters(parameters, settings, 1.0)


@pytest.mark.parametrize(
    ("manual_factor", "autoscale", "expected"),
    [(None, True, 1.0330008543), (2.0, True, 2.0), (None, False, 1.0)],
)
def test_nsb_rate_factor(manual_factor, autoscale, expected):
    parameters = {"nsb_autoscale_airmass": {"value": "0.84 0.29"}}
    settings = NsbSettings(zenith_angle=20.0, manual_factor=manual_factor, autoscale=autoscale)

    assert nsb_rate_factor(parameters, settings, 1.0) == pytest.approx(expected)
