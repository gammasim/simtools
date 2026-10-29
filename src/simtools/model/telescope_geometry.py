"""Nominal optical geometry derived from telescope model data."""

import math
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy.table import Table

SHAPES = {0: "circle", 1: "hexagon_flat_x", 2: "square", 3: "hexagon_flat_y"}

STRUCTURE_FRAME = "primary_mirror_frame"

CIRCLE_CORNERS = 48


@dataclass(frozen=True)
class Element:
    """Element of the telescope in its frame, with positions in metres.

    A cylinder has two points (the end points of its axis) and a diameter. A polygon is
    given by its corners (a mirror aperture, a camera plate, or the corners of a
    structure element whose convex hull is drawn). A wire is drawn as its polylines.
    """

    name: str
    group: str
    points: np.ndarray
    diameter: float = 0.0
    kind: str = "polygon"  # "polygon", or "wire": polylines, separated by rows of nan.
    side_only: bool = False  # Drawn in the side view only (e.g. a profile).

    @property
    def is_cylinder(self):
        """Whether the element is a cylinder."""
        return self.diameter > 0.0 and len(self.points) == 2


@dataclass
class TelescopeGeometry:
    """Elements to draw: mirrors, camera housing and telescope structure."""

    name: str
    mirrors: list = field(default_factory=list)
    camera: list = field(default_factory=list)
    structure: list = field(default_factory=list)


def _in_metres(value, unit):
    """Return a length (or a list of lengths with one unit per value) in metres."""
    if isinstance(unit, list):
        return np.array(
            [u.Quantity(v, un).to_value(u.m) for v, un in zip(value, unit, strict=True)]
        )
    return u.Quantity(value, unit).to_value(u.m)


def _metres(column):
    """Column (of a table with units) as an array in metres."""
    return u.Quantity(column).to_value(u.m)


def _parameter(parameters, name, default=None):
    """Value of a model parameter in metres, or the plain value if it has no length unit."""
    record = parameters.get(name)
    if record is None or record.get("value") is None:
        return default
    unit = record.get("unit")
    if unit in (None, "", "None", "dimensionless"):
        return record["value"]
    return _in_metres(record["value"], unit)


def _aperture_corners(shape, half_width):
    """Corners (n, 2) of an aperture in its (u, v) frame; hexagon flats at +-half_width."""
    h = half_width
    if shape == "circle":
        angles = np.linspace(0.0, 2 * np.pi, CIRCLE_CORNERS, endpoint=False)
        return h * np.column_stack([np.cos(angles), np.sin(angles)])
    if shape == "square":
        return np.array([(h, h), (-h, h), (-h, -h), (h, -h)])
    r = 2 * h / math.sqrt(3)  # Distance of a hexagon corner from its centre.
    if shape == "hexagon_flat_x":
        return np.array([(0, r), (h, r / 2), (h, -r / 2), (0, -r), (-h, -r / 2), (-h, r / 2)])
    if shape == "hexagon_flat_y":
        return np.array([(r, 0), (r / 2, h), (-r / 2, h), (-r, 0), (-r / 2, -h), (r / 2, -h)])
    raise ValueError(f"unsupported aperture shape {shape}")


def _plane_corners(centre, normal, tangent, shape, half_width):
    """Corners (n, 3) of an aperture in the plane through centre (frame as in obdeect)."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    u_axis = np.asarray(tangent, dtype=float)
    u_axis = u_axis - np.dot(u_axis, n) * n
    u_axis = u_axis / np.linalg.norm(u_axis)
    v_axis = np.cross(n, u_axis)
    local = _aperture_corners(shape, half_width)
    return np.asarray(centre, dtype=float) + local[:, :1] * u_axis + local[:, 1:] * v_axis


def _nominal_placement(x, y, z_file, focal, dish, parabolic, opt):
    """Nominal height and inclination of a mirror on the dish (sim_telarray's placement)."""
    radius = math.hypot(x, y)
    if z_file:
        height = abs(z_file)
    elif parabolic:
        height = radius**2 / (4 * dish)
    else:
        if radius > dish:
            raise ValueError("a mirror lies outside the dish sphere")
        height = dish - math.sqrt(dish * dish - radius * radius)
    distance = math.hypot(focal - height, radius) * (1 + opt[2] + opt[1] * radius**2 / focal**2)
    z = z_file if z_file > 0 else focal - math.sqrt(distance * distance - radius * radius)
    inclination = 0.5 * math.asin(radius / distance) * (1 + opt[0] * radius**2 / focal**2)
    return z, inclination


def _mirror_facets(parameters, mirror_table, focal, overall):
    """Facets of a segmented dish: the mirror list at its nominal positions."""
    dish = _parameter(parameters, "dish_shape_length", 0.0) or focal
    parabolic = bool(parameters.get("parabolic_dish", {}).get("value", False))
    opt = [float(v) for v in parameters.get("mirror_opt", {}).get("value", [0.0, 0.0, 0.0])]
    flip = bool(parameters.get("flip_mirrors", {}).get("value", False))
    x_all = _metres(mirror_table["mirror_x"])
    y_all = _metres(mirror_table["mirror_y"])
    z_all = _metres(mirror_table["mirror_z"]) if "mirror_z" in mirror_table.colnames else 0.0
    z_all = np.broadcast_to(z_all, x_all.shape)
    diameter_all = _metres(mirror_table["mirror_diameter"])
    facets = []
    for i, (x, y, z_file, diameter, code) in enumerate(
        zip(x_all, y_all, z_all, diameter_all, mirror_table["shape_type"], strict=True)
    ):
        shape = SHAPES[int(code)]
        if flip:
            x, y = y, x
            shape = {"hexagon_flat_x": "hexagon_flat_y", "hexagon_flat_y": "hexagon_flat_x"}.get(
                shape, shape
            )
        z, inclination = _nominal_placement(x, y, z_file, focal, dish, parabolic, opt)
        phi = math.atan2(y, x)
        c, s = math.cos(phi), math.sin(phi)
        ci, si = math.cos(inclination), math.sin(inclination)
        normal = [-c * si, -s * si, ci]
        tangent = [c * c * ci + s * s, c * s * ci - c * s, c * si]
        facets.append(
            Element(
                name=f"mirror-{i:03d}",
                group="mirror",
                points=_plane_corners([x, y, z - overall], normal, tangent, shape, 0.5 * diameter),
            )
        )
    return facets


def _asphere_coefficients(parameters, prefix):
    """Coefficients (m) and radial scale (m) of an asphere, as in obdeect and sim_config.c.

    The sag is z(r) = sum_i c_i (r / scale)^(2 i). Without a reference radius, the
    coefficients are powers of centimetres.
    """
    record = parameters[f"{prefix}_parameters"]
    values = record["value"]
    if f"{prefix}_ref_radius" in parameters:
        scale = _parameter(parameters, f"{prefix}_ref_radius")
        return [float(v) * scale for v in values], scale
    return [float(v) * 0.01 ** (1 - 2 * i) for i, v in enumerate(values)], 1.0


def _secondary_coefficients(parameters):
    """Secondary coefficients in the telescope frame, as in obdeect and sim_telarray.

    sim_telarray rotates the secondary frame by pi about y: the vertex changes sign if it
    lies below the primary's, and the polynomial terms always change sign.
    """
    primary, _ = _asphere_coefficients(parameters, "primary_mirror")
    coefficients, scale = _asphere_coefficients(parameters, "secondary_mirror")
    if coefficients[0] < primary[0]:
        coefficients[0] = -coefficients[0]
    coefficients[1:] = [-c for c in coefficients[1:]]
    return coefficients, scale


def _sag(coefficients, scale, radius):
    """Sag z(r) of an asphere in metres, at radii in metres (vertex included)."""
    q = (np.asarray(radius, dtype=float) / scale) ** 2
    return sum(c * q**i for i, c in enumerate(coefficients))


def _dual_mirror_apertures(parameters, overall):
    """Dual-mirror surfaces: rim discs, sag profiles and holes, in the telescope frame.

    The rim of each mirror is a disc at the sag height of its rim; its profile is drawn in
    the side view, where the curvature shows.
    """
    apertures = []
    for prefix in ("primary_mirror", "secondary_mirror"):
        name = prefix.split("_", maxsplit=1)[0]
        if prefix == "secondary_mirror":
            coefficients, scale = _secondary_coefficients(parameters)
        else:
            coefficients, scale = _asphere_coefficients(parameters, prefix)
        outer = 0.5 * _parameter(parameters, f"{prefix}_diameter")
        hole = 0.5 * (_parameter(parameters, f"{prefix}_hole_diameter", 0.0) or 0.0)
        radii = np.linspace(hole, outer, 40)
        profile = []
        for theta in (0.0, np.pi):
            profile.append(
                np.column_stack(
                    [
                        radii * np.cos(theta),
                        radii * np.sin(theta),
                        _sag(coefficients, scale, radii) - overall,
                    ]
                )
            )
            profile.append(np.full((1, 3), np.nan))
        apertures.append(
            Element(
                name=name,
                group="mirror",
                points=_plane_corners(
                    [0.0, 0.0, _sag(coefficients, scale, outer) - overall],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    "circle",
                    outer,
                ),
            )
        )
        apertures.append(
            Element(
                name=f"{name}-profile",
                group="mirror",
                points=np.vstack(profile),
                kind="wire",
                side_only=True,
            )
        )
        if hole > 0.0:
            apertures.append(
                Element(
                    name=f"{name}-hole",
                    group="mirror_hole",
                    points=_plane_corners(
                        [0.0, 0.0, _sag(coefficients, scale, hole) - overall],
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0],
                        "circle",
                        hole,
                    ),
                )
            )
    return apertures


def _circle(z, radius):
    """Circle (48 points) of a given radius in the plane at height z."""
    return _plane_corners([0.0, 0.0, z], [0, 0, 1], [1, 0, 0], "circle", radius)


def _secondary_baffle(parameters, overall):
    """Secondary baffle (open cone wall) as a wire: its rings and generators.

    The values are z1, z2, R1, the wall thickness and R2 (zero: R2 = R1), as in obdeect.
    """
    record = parameters.get("secondary_mirror_baffle")
    if not record or not record.get("value"):
        return []
    values = list(record["value"]) + [0.0] * (5 - len(record["value"]))
    unit = record["unit"][0] if isinstance(record["unit"], list) else record["unit"]
    z1, z2, r1, thickness, r2 = _in_metres(values, unit)
    if r1 <= 0.0:
        return []
    r2 = r2 or r1
    if thickness >= r1 > 0.0:
        thickness -= r1
    z1, z2 = z1 - overall, z2 - overall
    rings, generators = [], []
    for radii in ((r1, r2), (r1 - thickness, r2 - thickness)):
        for z, radius in zip((z1, z2), radii, strict=True):
            rings.append(_circle(z, radius))
            rings.append(np.full((1, 3), np.nan))
    for theta in (0.0, np.pi):
        for radii in ((r1, r2), (r1 - thickness, r2 - thickness)):
            generators.append(
                np.array(
                    [
                        [radii[0] * np.cos(theta), radii[0] * np.sin(theta), z1],
                        [radii[1] * np.cos(theta), radii[1] * np.sin(theta), z2],
                        [np.nan] * 3,
                    ]
                )
            )
    return [
        Element(name="secondary-baffle", group="secondary", points=np.vstack(rings), kind="wire"),
        Element(
            name="secondary-baffle",
            group="secondary",
            points=np.vstack(generators),
            kind="wire",
            side_only=True,
        ),
    ]


def _secondary_shadow(parameters, overall):
    """Secondary shadow plate (as in obdeect): a disc that stops the light before the mirror."""
    keys = ("secondary_mirror_shadow_diameter", "secondary_mirror_shadow_offset")
    if not all(key in parameters for key in keys):
        return []
    diameter = _parameter(parameters, keys[0])
    coefficients, scale = _secondary_coefficients(parameters)
    if diameter < 0:
        diameter = 2 * 0.5 * _parameter(parameters, "secondary_mirror_diameter")
    if diameter <= 0:
        return []
    offset = _parameter(parameters, keys[1])
    if offset <= 0:  # Concave secondary: the plate at its rim, otherwise at its vertex.
        offset = _sag(coefficients, scale, 0.5 * diameter)
        if coefficients[1] >= 0:
            offset = coefficients[0]
    else:
        offset -= overall
    return [
        Element(name="secondary-shadow", group="secondary", points=_circle(offset, 0.5 * diameter))
    ]


def _camera_housing(parameters, front, dual):
    """Camera body: the front plate, and the back plate of its depth (camera_depth)."""
    diameter = _parameter(parameters, "camera_body_diameter", 0.0)
    if not diameter:
        return []
    shape_code = parameters.get("camera_body_shape", {}).get("value", 0)
    shape = SHAPES.get(int(shape_code))
    if shape is None:
        raise ValueError(f"unsupported camera_body_shape {shape_code}")
    depth = _parameter(parameters, "camera_depth", 0.0) or 0.0
    # The body extends from the front plate towards the mirrors for dual-mirror
    # telescopes (sim_telarray's camera body), and away from them otherwise.
    heights = [front - depth, front] if dual and depth > 0 else [front, front + depth]
    heights = sorted(set(heights))
    corners = [
        _plane_corners([0.0, 0.0, z], [0, 0, 1], [1, 0, 0], shape, 0.5 * diameter) for z in heights
    ]
    return [Element(name="camera-housing", group="camera", points=np.vstack(corners))]


def _structure_elements(structure_file, overall):
    """Telescope structure (ECSV table, primary mirror frame, metres) as elements."""
    table = Table.read(structure_file, format="ascii.ecsv")
    frame = table.meta.get("coordinate_frame")
    if frame != STRUCTURE_FRAME:
        raise ValueError(f"telescope_structure has coordinate_frame {frame}, not {STRUCTURE_FRAME}")
    corners = sum(1 for c in table.colnames if c[0] == "x" and c[1:].isdigit())
    coordinates = {
        f"{axis}{i}": _metres(table[f"{axis}{i}"]) for i in range(1, corners + 1) for axis in "xyz"
    }
    diameters = _metres(table["diameter"]) if "diameter" in table.colnames else None
    elements = []
    for index, row in enumerate(table):
        points = []
        for i in range(1, corners + 1):
            x = coordinates[f"x{i}"][index]
            if not np.isnan(x):
                points.append(
                    [x, coordinates[f"y{i}"][index], coordinates[f"z{i}"][index] - overall]
                )
        group = str(row["group"]) if "group" in table.colnames else "structure"
        diameter = float(diameters[index]) if diameters is not None else 0.0
        elements.append(
            Element(
                name=str(row["id"]),
                group=group,
                points=np.array(points),
                diameter=0.0 if np.isnan(diameter) else diameter,
            )
        )
    return elements


def telescope_geometry(tel_model):
    """Nominal geometry of a telescope model, for plotting.

    Parameters
    ----------
    tel_model: TelescopeModel
        Telescope model with exported model files (``export_model_files``).

    Returns
    -------
    TelescopeGeometry
        Mirrors (facets, or the apertures of a dual-mirror telescope), camera housing,
        and telescope structure (empty if the model has none).
    """
    parameters = tel_model.parameters
    overall = _parameter(parameters, "mirror_offset", 0.0)
    dual = "primary_mirror_parameters" in parameters
    if dual:
        mirrors = _dual_mirror_apertures(parameters, overall)
        front = _asphere_coefficients(parameters, "focal_surface")[0][0] - overall
    else:
        focal = _parameter(parameters, "focal_length")
        mirrors = _mirror_facets(parameters, tel_model.mirrors.mirror_table, focal, overall)
        front = focal - overall
    structure = []
    if dual:
        structure += _secondary_baffle(parameters, overall)
        structure += _secondary_shadow(parameters, overall)
    record = parameters.get("telescope_structure")
    if record and record.get("value"):
        structure += _structure_elements(tel_model.config_file_directory / record["value"], overall)
    return TelescopeGeometry(
        name=tel_model.name,
        mirrors=mirrors,
        camera=_camera_housing(parameters, front, dual),
        structure=structure,
    )
