"""Tests for model-based telescope sphere bounds."""

from unittest.mock import Mock

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable
from astropy.tests.helper import assert_quantity_allclose

from simtools.model.model_parameter import InvalidModelParameterError
from simtools.model.telescope_sphere import derive_telescope_sphere_radius

_PRIMARY_MIRROR_REF_RADIUS = "primary_mirror_ref_radius"
_PRIMARY_MIRROR_PARAMETERS = "primary_mirror_parameters"
_MIRROR_X = "mirror_x"
_FINITE = "finite"
_FOCAL_LENGTH = "focal_length"
_DISH_SHAPE_LENGTH = "dish_shape_length"
_MIRROR_FOCAL_LENGTH = "mirror_focal_length"
_SHAPE_TYPE = "shape_type"
_MIRROR_Z = "mirror_z"


def _model(mirror_class=0):
    model = Mock()
    model.get_parameter_value.side_effect = lambda name: (
        mirror_class if name == "mirror_class" else 0
    )
    values = {
        "axes_offsets": [0, 0] * u.m,
        "mirror_offset": 0 * u.m,
        _FOCAL_LENGTH: 5 * u.m,
        _DISH_SHAPE_LENGTH: 5 * u.m,
        _MIRROR_FOCAL_LENGTH: 5 * u.m,
        "primary_mirror_diameter": 4 * u.m,
        "secondary_mirror_diameter": 2 * u.m,
        _PRIMARY_MIRROR_REF_RADIUS: 1 * u.m,
        "secondary_mirror_ref_radius": 1 * u.m,
        _PRIMARY_MIRROR_PARAMETERS: [0, 0.25] * u.m,
        "secondary_mirror_parameters": [-4, 0] * u.m,
    }
    model.get_parameter_value_with_unit.side_effect = values.__getitem__
    model.get_parameter_table.return_value = QTable(
        {
            _MIRROR_X: [1] * u.m,
            "mirror_y": [0] * u.m,
            "mirror_diameter": [1] * u.m,
            _MIRROR_Z: [0] * u.m,
            _SHAPE_TYPE: [1],
            _FOCAL_LENGTH: [5] * u.m,
        }
    )
    return model


def test_single_mirror_encloses_panel_corners_and_sag():
    result = derive_telescope_sphere_radius(_model())
    assert result.to_value(u.m) == pytest.approx(1.59, rel=0.01)


def test_single_piece_mirror_uses_panel_bounds():
    assert_quantity_allclose(
        derive_telescope_sphere_radius(_model(1)), derive_telescope_sphere_radius(_model())
    )


def test_camera_does_not_enlarge_corsika_entrance_sphere():
    model = _model()
    assert derive_telescope_sphere_radius(model) < 2 * u.m
    assert "camera_body_shape" not in [
        call.args[0] for call in model.get_parameter_value.call_args_list
    ]


def test_square_panel_corners_are_enclosed():
    model = _model()
    model.get_parameter_table.return_value[_SHAPE_TYPE] = [2]
    assert derive_telescope_sphere_radius(model) > derive_telescope_sphere_radius(_model())


def test_dual_mirror_encloses_polynomial_surfaces():
    result = derive_telescope_sphere_radius(_model(2))
    assert_quantity_allclose(result, np.hypot(1, 4) * u.m)


def test_polynomial_coefficients_are_not_scaled_by_reference_radius():
    from simtools.model.telescope_sphere import _polynomial_bounds

    model = _model(2)
    values = model.get_parameter_value_with_unit.side_effect
    model.get_parameter_value_with_unit.side_effect = lambda name: (
        2 * u.m
        if name == _PRIMARY_MIRROR_REF_RADIUS
        else [0.25, 0] * u.m
        if name == _PRIMARY_MIRROR_PARAMETERS
        else values(name)
    )
    assert _polynomial_bounds(model, "primary", 1) == pytest.approx((0.25, 0.25))


def test_missing_geometry_is_actionable():
    model = _model()
    model.get_parameter_table.return_value = QTable()
    with pytest.raises(ValueError, match="mirror_list needs positions"):
        derive_telescope_sphere_radius(model)


def test_missing_mirror_table_is_actionable():
    model = _model()
    model.get_parameter_table.side_effect = InvalidModelParameterError("missing mirror_list")
    with pytest.raises(ValueError, match="mirror_list table is required"):
        derive_telescope_sphere_radius(model)


def test_missing_length_parameter_is_actionable():
    model = _model()
    model.get_parameter_value_with_unit.side_effect = InvalidModelParameterError(
        "missing focal_length"
    )
    with pytest.raises(ValueError, match="focal_length"):
        derive_telescope_sphere_radius(model)


def test_invalid_mirror_class():
    with pytest.raises(ValueError, match="Unsupported mirror_class"):
        derive_telescope_sphere_radius(_model(3))


def test_nonfinite_dish_length():
    model = _model()
    values = model.get_parameter_value_with_unit.side_effect
    model.get_parameter_value_with_unit.side_effect = lambda name: (
        np.nan * u.m if name == _DISH_SHAPE_LENGTH else values(name)
    )
    with pytest.raises(ValueError, match=_FINITE):
        derive_telescope_sphere_radius(model)


def test_missing_dish_length():
    model = _model()
    values = model.get_parameter_value_with_unit.side_effect

    def read_length(name):
        if name == _DISH_SHAPE_LENGTH:
            raise KeyError(name)
        return values(name)

    model.get_parameter_value_with_unit.side_effect = read_length
    with pytest.raises(ValueError, match=_DISH_SHAPE_LENGTH):
        derive_telescope_sphere_radius(model)


def test_nonfinite_panel_geometry():
    model = _model()
    model.get_parameter_table.return_value[_MIRROR_X] = [np.nan] * u.m
    with pytest.raises(ValueError, match=_FINITE):
        derive_telescope_sphere_radius(model)


def test_parabolic_dish_uses_parabolic_panel_heights():
    from simtools.model.telescope_sphere import _panel_heights

    model = _model()
    model.get_parameter_value.side_effect = lambda name: name == "parabolic_dish"
    height = _panel_heights(model, np.array([2.0]), np.array([0.0]))
    assert height[0] == pytest.approx(0.2)


def test_invalid_davies_cotton_dish_length():
    model = _model()
    values = model.get_parameter_value_with_unit.side_effect
    model.get_parameter_value_with_unit.side_effect = lambda name: (
        0.5 * u.m if name == _DISH_SHAPE_LENGTH else values(name)
    )
    with pytest.raises(ValueError, match="Davies-Cotton"):
        derive_telescope_sphere_radius(model)


def test_default_panel_focal_length():
    model = _model()
    model.get_parameter_table.return_value[_FOCAL_LENGTH] = [0] * u.m
    assert derive_telescope_sphere_radius(model) < 2 * u.m


def test_invalid_panel_curvature():
    model = _model()
    model.get_parameter_table.return_value[_FOCAL_LENGTH] = [0.1] * u.m
    with pytest.raises(ValueError, match="panel focal lengths"):
        derive_telescope_sphere_radius(model)


def test_panel_at_focus_has_invalid_orientation():
    model = _model()
    model.get_parameter_table.return_value[_MIRROR_X] = [0] * u.m
    model.get_parameter_table.return_value[_MIRROR_Z] = [5] * u.m
    with pytest.raises(ValueError, match="distance to focus"):
        derive_telescope_sphere_radius(model)


def test_nonfinite_panel_shape():
    model = _model()
    model.get_parameter_table.return_value[_SHAPE_TYPE] = [np.nan]
    with pytest.raises(ValueError, match="shapes must be finite"):
        derive_telescope_sphere_radius(model)


def test_nonfinite_panel_focal_length():
    model = _model()
    model.get_parameter_table.return_value[_FOCAL_LENGTH] = [np.nan] * u.m
    with pytest.raises(ValueError, match="focal lengths must be finite"):
        derive_telescope_sphere_radius(model)


def test_invalid_polynomial_reference_radius():
    model = _model(2)
    values = model.get_parameter_value_with_unit.side_effect
    model.get_parameter_value_with_unit.side_effect = lambda name: (
        0 * u.m if name == _PRIMARY_MIRROR_REF_RADIUS else values(name)
    )
    with pytest.raises(ValueError, match="reference radius"):
        derive_telescope_sphere_radius(model)


def test_missing_polynomial_coefficients():
    model = _model(2)
    values = model.get_parameter_value_with_unit.side_effect
    model.get_parameter_value_with_unit.side_effect = lambda name: (
        None if name == _PRIMARY_MIRROR_PARAMETERS else values(name)
    )
    with pytest.raises(ValueError, match="coefficients are required"):
        derive_telescope_sphere_radius(model)


def test_nonfinite_polynomial_coefficients():
    model = _model(2)
    values = model.get_parameter_value_with_unit.side_effect
    model.get_parameter_value_with_unit.side_effect = lambda name: (
        [0, np.nan] * u.m if name == _PRIMARY_MIRROR_PARAMETERS else values(name)
    )
    with pytest.raises(ValueError, match="finite coefficients"):
        derive_telescope_sphere_radius(model)


def test_axes_offsets_are_normalized_to_quantity_array():
    model = _model()
    values = model.get_parameter_value_with_unit.side_effect
    model.get_parameter_value_with_unit.side_effect = lambda name: (
        [160 * u.cm, 0 * u.cm] if name == "axes_offsets" else values(name)
    )
    result = derive_telescope_sphere_radius(model)
    assert result.unit == u.m
