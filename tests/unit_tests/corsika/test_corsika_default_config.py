import astropy.units as u
import pytest

from simtools.corsika.corsika_default_config import CorsikaDefaultConfig


def compare_lists_ignoring_units(list1, list2):
    return all(list1[i].value == pytest.approx(list2[i]) for i in range(len(list1)))


def test_init():
    config = CorsikaDefaultConfig(primary="gamma", zenith_angle=20.0 * u.deg)
    assert config.primary == "gamma"
    assert config.zenith_angle.value == pytest.approx(20.0)


def test_invalid_init(caplog):
    with pytest.raises(ValueError):
        CorsikaDefaultConfig(primary="invalid_particle", zenith_angle=20.0 * u.deg)
        assert "Invalid primary particle: invalid_particle" in caplog.text


def test_primary_setter():
    config = CorsikaDefaultConfig(primary="gamma", zenith_angle=20.0 * u.deg)
    config.primary = "proton"
    assert config.primary == "proton"


def test_primary_setter_invalid(caplog):
    config = CorsikaDefaultConfig(primary="gamma", zenith_angle=20.0 * u.deg)
    with pytest.raises(ValueError):
        config.primary = "invalid_particle"
        assert "Invalid primary particle: invalid_particle" in caplog.text


def test_zenith_angle_setter():
    config = CorsikaDefaultConfig(primary="gamma", zenith_angle=20.0 * u.deg)
    config.zenith_angle = 30.0 * u.deg
    assert config.zenith_angle.value == pytest.approx(30.0)


def test_zenith_angle_setter_invalid(caplog):
    with pytest.raises(ValueError):
        config = CorsikaDefaultConfig(primary="gamma", zenith_angle=20.0 * u.deg)
        config.zenith_angle = 5.0 * u.deg
        assert "outside of the allowed interval" in caplog.text


def test_energy_range_for_primary():
    config = CorsikaDefaultConfig(primary="gamma", zenith_angle=40.0 * u.deg)
    energy_range = config.energy_range_for_primary()
    expected_energies = [6.0, 660.0]
    compare_lists_ignoring_units(energy_range, expected_energies)

    config = CorsikaDefaultConfig(primary="gamma", zenith_angle=52.0 * u.deg)
    energy_range = config.energy_range_for_primary()
    expected_energies = [9.24, 858.0]
    compare_lists_ignoring_units(energy_range, expected_energies)


def test_number_of_showers_for_primary():
    config = CorsikaDefaultConfig(primary="gamma", zenith_angle=40.0 * u.deg)
    number_of_showers = config.number_of_showers_for_primary()
    assert number_of_showers == 5000


def test_view_cone_for_primary():
    config = CorsikaDefaultConfig(primary="gamma", zenith_angle=20.0 * u.deg)
    view_cone = config.view_cone_for_primary()
    expected_view_cone = [0.0, 0.0]
    compare_lists_ignoring_units(view_cone, expected_view_cone)

    config = CorsikaDefaultConfig(primary="gamma_diffuse", zenith_angle=20.0 * u.deg)
    view_cone = config.view_cone_for_primary()
    expected_view_cone = [0.0, 10.0]
    compare_lists_ignoring_units(view_cone, expected_view_cone)

    config = CorsikaDefaultConfig(primary="proton", zenith_angle=20.0 * u.deg)
    view_cone = config.view_cone_for_primary()
    expected_view_cone = [0.0, 10.0]
    compare_lists_ignoring_units(view_cone, expected_view_cone)


def test_interpolate_to_zenith_angle():
    config = CorsikaDefaultConfig(primary="gamma", zenith_angle=20.0 * u.deg)

    # Test case 1: Interpolate energy range
    zenith_angles_to_interpolate = [30.0, 40.0, 50.0]
    values_to_interpolate = [6.0, 9.24, 12.5]
    interpolated_value = config.interpolate_to_zenith_angle(
        35.0 * u.deg, zenith_angles_to_interpolate, values_to_interpolate
    )
    assert interpolated_value == pytest.approx(7.6175)

    # Test case 2: Interpolate number of showers
    zenith_angles_to_interpolate = [30.0, 40.0, 50.0]
    values_to_interpolate = [5000, 6000, 7000]
    interpolated_value = config.interpolate_to_zenith_angle(
        45.0 * u.deg, zenith_angles_to_interpolate, values_to_interpolate
    )
    assert interpolated_value == pytest.approx(6500)


def test_energy_slope_getter():
    config = CorsikaDefaultConfig(primary="gamma", zenith_angle=20.0 * u.deg)
    assert config.energy_slope == pytest.approx(-2.0)
