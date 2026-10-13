"""Derive a CORSIKA enclosure from the main optical telescope elements."""

import astropy.units as u
import numpy as np

from simtools.model.model_parameter import InvalidModelParameterError
from simtools.utils.geometry import fiducial_radius_from_shape, telescope_fiducial_sphere_radius

_PRIMARY_DIAMETER = "primary_mirror_diameter"
_SECONDARY_DIAMETER = "secondary_mirror_diameter"
_SECONDARY = "secondary"
_FOCAL_LENGTH = "focal_length"
_DISH_SHAPE_LENGTH = "dish_shape_length"
_MIRROR_FOCAL_LENGTH = "mirror_focal_length"
_MIRROR_LIST = "mirror_list"
_MIRROR_X = "mirror_x"
_MIRROR_Y = "mirror_y"
_MIRROR_DIAMETER = "mirror_diameter"
_MIRROR_Z = "mirror_z"
_PARABOLIC_DISH = "parabolic_dish"
_SHAPE_TYPE = "shape_type"


def _length(model, name):
    """Read a model length in metres with a useful error for absent values."""
    try:
        value = model.get_parameter_value_with_unit(name)
        return float(value.to_value(u.m))
    except (AttributeError, InvalidModelParameterError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"A length-valued {name} is required to derive the sphere.") from exc


def _polynomial_bounds(model, surface, radius):
    """Bound a rotational mirror polynomial at its stationary points and rim."""
    name = f"{surface}_mirror_parameters"
    reference = _length(model, f"{surface}_mirror_ref_radius")
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError(f"{surface} mirror reference radius must be positive and finite.")
    try:
        coefficients = np.asarray(
            u.Quantity(model.get_parameter_value_with_unit(name)).to_value(u.m)
        )
    except (AttributeError, InvalidModelParameterError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Length-valued {name} coefficients are required.") from exc
    if coefficients.ndim != 1 or coefficients.size == 0 or not np.all(np.isfinite(coefficients)):
        raise ValueError(f"{name} must contain finite coefficients.")
    # The secondary local optical axis points toward the primary.
    sign = -1 if surface == _SECONDARY else 1
    polynomial = np.polynomial.Polynomial(coefficients)
    critical = polynomial.deriv().roots()
    upper = (radius / reference) ** 2
    points = [0.0, upper]
    points.extend(
        float(root.real) for root in critical if abs(root.imag) < 1e-10 and 0 < root.real < upper
    )
    values = sign * polynomial(points)
    return float(np.min(values)), float(np.max(values))


def _panel_heights(model, radius, listed_z):
    """Calculate nominal panel-centre heights as sim_telarray does."""
    focal_length = _length(model, _FOCAL_LENGTH)
    dish_length = _length(model, _DISH_SHAPE_LENGTH) or focal_length
    if not np.isfinite(dish_length) or dish_length <= 0:
        raise ValueError("Dish shape length must be positive and finite.")
    try:
        parabolic_dish = model.get_parameter_value(_PARABOLIC_DISH)
    except InvalidModelParameterError as exc:
        raise ValueError(f"A model parameter {_PARABOLIC_DISH} is required.") from exc
    if parabolic_dish:
        height = radius**2 / (4 * dish_length)
    else:
        if dish_length <= np.max(radius):
            raise ValueError("Davies-Cotton dish length must exceed panel-centre radii.")
        height = dish_length - np.sqrt(dish_length**2 - radius**2)
    return np.where(listed_z != 0, listed_z, height)


def _panel_extent(model, centre_radius, panel_radius, height, focal_length):
    """Bound axial tilt and spherical sag of each panel."""
    nominal_focus = _length(model, _FOCAL_LENGTH)
    if np.any(focal_length <= 0):
        default_focal_length = _length(model, _MIRROR_FOCAL_LENGTH) or nominal_focus
        focal_length = np.where(focal_length > 0, focal_length, default_focal_length)
    if np.any(focal_length <= 0) or np.any(2 * focal_length <= panel_radius):
        raise ValueError("Mirror panel focal lengths must exceed half the panel radius.")
    distance = np.hypot(nominal_focus - height, centre_radius)
    if np.any(distance <= 0):
        raise ValueError("Mirror panel distance to focus must be positive.")
    inclination = np.arcsin(centre_radius / distance) / 2
    sag = 2 * focal_length - np.sqrt((2 * focal_length) ** 2 - panel_radius**2)
    return panel_radius * np.sin(inclination) + sag


def _single_mirror_bounds(model):
    """Bound mirror panels using their dish height, corners, tilt, and sag."""
    try:
        table = model.get_parameter_table(_MIRROR_LIST)
    except InvalidModelParameterError as exc:
        raise ValueError("A mirror_list table is required to derive the sphere.") from exc
    needed = (
        _MIRROR_X,
        _MIRROR_Y,
        _MIRROR_DIAMETER,
        _MIRROR_Z,
        _SHAPE_TYPE,
        _FOCAL_LENGTH,
    )
    if len(table) == 0 or any(column not in table.colnames for column in needed):
        raise ValueError("mirror_list needs positions, diameters, shapes, and focal lengths.")
    x = np.asarray(u.Quantity(table[_MIRROR_X]).to_value(u.m))
    y = np.asarray(u.Quantity(table[_MIRROR_Y]).to_value(u.m))
    diameter = np.asarray(u.Quantity(table[_MIRROR_DIAMETER]).to_value(u.m))
    z = np.asarray(u.Quantity(table[_MIRROR_Z]).to_value(u.m))
    coordinates = (x, y, diameter, z)
    if not all(np.all(np.isfinite(column)) for column in coordinates) or np.any(diameter <= 0):
        raise ValueError("mirror_list positions and positive panel diameters must be finite.")
    shape = np.asarray(table[_SHAPE_TYPE])
    if not np.all(np.isfinite(shape)):
        raise ValueError("Mirror panel shapes must be finite.")
    panel_radius = np.array(
        [fiducial_radius_from_shape(width, kind) for width, kind in zip(diameter, shape)]
    )
    centre_radius = np.hypot(x, y)
    height = _panel_heights(model, centre_radius, z)
    focal_length = u.Quantity(table[_FOCAL_LENGTH]).to_value(u.m)
    if not np.all(np.isfinite(focal_length)):
        raise ValueError("Mirror panel focal lengths must be finite.")
    clearance = _panel_extent(model, centre_radius, panel_radius, height, focal_length)
    radius = float(np.max(centre_radius + panel_radius))
    return radius, (float(np.min(height - clearance)), float(np.max(height + clearance)))


def derive_telescope_sphere_radius(model, zenith_range=(0, 90) * u.deg, margin=0 * u.m):
    """Return a conservative CORSIKA entrance sphere radius for the model's mirrors.

    Parameters
    ----------
    model : TelescopeModel
        Simulation model with primary and optional secondary reflector geometry.
    zenith_range : astropy.units.Quantity, optional
        Minimum and maximum telescope zenith angles.
    margin : astropy.units.Quantity, optional
        Extra radial clearance.

    Returns
    -------
    astropy.units.Quantity
        Fiducial sphere radius in metres.

    Raises
    ------
    ValueError
        If required model geometry is missing or invalid.
    """
    try:
        mirror_class = model.get_parameter_value("mirror_class")
    except InvalidModelParameterError as exc:
        raise ValueError("A mirror_class parameter is required to derive the sphere.") from exc
    if mirror_class in (0, 1):
        primary = _single_mirror_bounds(model)
        elements = [primary]
    elif mirror_class == 2:
        elements = []
        for surface, diameter_name in (
            ("primary", _PRIMARY_DIAMETER),
            (_SECONDARY, _SECONDARY_DIAMETER),
        ):
            radius = _length(model, diameter_name) / 2
            elements.append((radius, _polynomial_bounds(model, surface, radius)))
    else:
        raise ValueError(f"Unsupported mirror_class {mirror_class}; expected 0, 1, or 2.")

    radii = np.array([radius for radius, _ in elements]) * u.m
    bounds = np.array([axial for _, axial in elements]) * u.m
    try:
        axes_offsets = u.Quantity(model.get_parameter_value_with_unit("axes_offsets")).to(u.m)
        mirror_offset = model.get_parameter_value_with_unit("mirror_offset")
    except (AttributeError, InvalidModelParameterError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Axes and mirror offsets are required to derive the sphere.") from exc
    return telescope_fiducial_sphere_radius(
        radii,
        bounds,
        axes_offsets,
        mirror_offset=mirror_offset,
        zenith_range=zenith_range,
        margin=margin,
    )
