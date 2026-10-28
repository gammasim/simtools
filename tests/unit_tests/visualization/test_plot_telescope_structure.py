"""Tests of the telescope structure plot (nominal geometry from model parameters)."""

import sys
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from simtools.visualization import plot_telescope_structure as pts

DISH = 19.2  # m, dish sphere of the segmented test telescope


def _record(value, unit=None):
    return {"value": value, "unit": unit}


def _segmented_stub(tmp_path=None, **extra):
    """Segmented telescope: two facets at x = 0 m and x = 1 m, camera 16 m above the dish."""
    parameters = {
        "focal_length": _record(1600.0, "cm"),
        "dish_shape_length": _record(1920.0, "cm"),
        "camera_body_diameter": _record(289.0, "cm"),
        "camera_body_shape": _record(2),
        "camera_depth": _record(0.0, "cm"),
    }
    parameters.update(extra)
    mirror_table = Table(
        {
            "mirror_x": u.Quantity([0.0, 100.0], u.cm),
            "mirror_y": u.Quantity([0.0, 0.0], u.cm),
            "mirror_z": u.Quantity([0.0, 0.0], u.cm),
            "mirror_diameter": u.Quantity([120.0, 120.0], u.cm),
            "focal_length": u.Quantity([1600.0, 1600.0], u.cm),
            "shape_type": [1.0, 0.0],
        }
    )
    return SimpleNamespace(
        name="TEST",
        parameters=parameters,
        config_file_directory=tmp_path,
        mirrors=SimpleNamespace(mirror_table=mirror_table),
    )


def _dual_stub(tmp_path=None, **extra):
    """Dual-mirror telescope with the SST-like parameters (cm)."""
    parameters = {
        "primary_mirror_parameters": _record([0.0, 6.08051e-4] + [0.0] * 18, ["cm"] * 20),
        "secondary_mirror_parameters": _record([310.84, 2.29358e-3] + [0.0] * 18, ["cm"] * 20),
        "focal_surface_parameters": _record([258.88, -4.71698e-3] + [0.0] * 18, ["cm"] * 20),
        "primary_mirror_diameter": _record(424.1, "cm"),
        "primary_mirror_hole_diameter": _record(0.0, "cm"),
        "secondary_mirror_diameter": _record(180.0, "cm"),
        "secondary_mirror_hole_diameter": _record(40.0, "cm"),
        "camera_body_diameter": _record(57.0, "cm"),
        "camera_body_shape": _record(2),
        "camera_depth": _record(50.0, "cm"),
    }
    parameters.update(extra)
    return SimpleNamespace(
        name="TEST-DUAL",
        parameters=parameters,
        config_file_directory=tmp_path,
        mirrors=SimpleNamespace(mirror_table=None),
    )


def test_facet_centres_follow_the_dish_sphere():
    geometry = pts.telescope_geometry(_segmented_stub())
    facets = geometry.mirrors
    assert len(facets) == 2
    centre_on_axis = facets[0].points.mean(axis=0)
    off_axis = facets[1].points.mean(axis=0)
    assert centre_on_axis == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    expected_height = DISH - np.sqrt(DISH**2 - 1.0)
    assert off_axis[:2] == pytest.approx([1.0, 0.0], abs=1e-9)
    assert off_axis[2] == pytest.approx(expected_height, abs=1e-9)


def test_facet_normal_bisects_sky_and_focus():
    """The nominal normal bisects the direction to the focus and the direction to the sky."""
    focal = 16.0
    geometry = pts.telescope_geometry(_segmented_stub())
    for facet in geometry.mirrors:
        centre = facet.points.mean(axis=0)
        normal = np.cross(facet.points[1] - facet.points[0], facet.points[2] - facet.points[0])
        normal = normal / np.linalg.norm(normal)
        if normal[2] < 0:
            normal = -normal
        to_focus = np.array([0.0, 0.0, focal]) - centre
        expected = to_focus / np.linalg.norm(to_focus) + np.array([0.0, 0.0, 1.0])
        expected = expected / np.linalg.norm(expected)
        assert normal == pytest.approx(expected, abs=2e-3)


def test_hexagon_facet_has_flats_and_six_corners():
    facet = pts.telescope_geometry(_segmented_stub()).mirrors[0]
    assert len(facet.points) == 6
    corner_distance = np.linalg.norm(facet.points - facet.points.mean(axis=0), axis=1)
    assert corner_distance.max() == pytest.approx(0.6 * 2 / np.sqrt(3), rel=1e-9)


def test_dual_mirror_surfaces_follow_the_asphere_sag():
    """Rims sit at the sag of the polynomial, z = c0 + c1 r^2 (c1 in m^-1 from cm)."""
    geometry = pts.telescope_geometry(_dual_stub())
    by_name = {element.name: element for element in geometry.mirrors}
    assert set(by_name) == {
        "primary",
        "primary-profile",
        "secondary",
        "secondary-profile",
        "secondary-hole",
    }
    primary_rim = 0.0608051 * 2.1205**2
    assert by_name["primary"].points[:, 2] == pytest.approx(primary_rim)
    secondary_rim = 3.1084 - 0.229358 * 0.9**2
    assert by_name["secondary"].points[:, 2] == pytest.approx(secondary_rim)
    assert np.linalg.norm(by_name["primary"].points[:, :2], axis=1).max() == pytest.approx(2.1205)
    hole = by_name["secondary-hole"]
    assert np.linalg.norm(hole.points[:, :2], axis=1).max() == pytest.approx(0.2)
    assert hole.points[0, 2] == pytest.approx(3.1084 - 0.229358 * 0.2**2)


def test_mirror_profiles_are_side_only_wires():
    geometry = pts.telescope_geometry(_dual_stub())
    profile = next(e for e in geometry.mirrors if e.name == "primary-profile")
    assert profile.kind == "wire"
    assert profile.side_only
    finite = profile.points[np.isfinite(profile.points[:, 0])]
    assert finite[:, 2].min() == pytest.approx(0.0)
    assert finite[:, 2].max() == pytest.approx(0.0608051 * 2.1205**2)


def test_secondary_baffle_is_a_wall_with_rings_and_generators():
    stub = _dual_stub(secondary_mirror_baffle=_record([300.0, 330.0, 100.0, 5.0, 95.0], "cm"))
    baffle = [e for e in pts.telescope_geometry(stub).structure if e.name == "secondary-baffle"]
    rings, generators = baffle
    assert rings.kind == "wire"
    assert not rings.side_only
    assert generators.side_only
    z = rings.points[:, 2]
    assert np.unique(np.round(z[np.isfinite(z)], 6)).tolist() == pytest.approx([3.0, 3.3])
    outer_at_z1 = rings.points[np.isclose(z, 3.0)]
    assert np.linalg.norm(outer_at_z1[:, :2], axis=1).max() == pytest.approx(1.0)


def test_secondary_shadow_plate_at_the_secondary_rim_by_default():
    stub = _dual_stub(
        secondary_mirror_shadow_diameter=_record(-1.0, "cm"),
        secondary_mirror_shadow_offset=_record(0.0, "cm"),
    )
    (plate,) = [e for e in pts.telescope_geometry(stub).structure if e.name == "secondary-shadow"]
    assert plate.points[:, 2] == pytest.approx(3.1084 - 0.229358 * 0.9**2)
    assert np.linalg.norm(plate.points[:, :2], axis=1).max() == pytest.approx(0.9)


def test_camera_housing_extends_away_from_the_mirrors_for_segmented_telescopes():
    stub = _segmented_stub(camera_depth=_record(50.0, "cm"))
    (housing,) = pts.telescope_geometry(stub).camera
    assert housing.points[:, 2].min() == pytest.approx(16.0)
    assert housing.points[:, 2].max() == pytest.approx(16.5)


def test_camera_housing_extends_towards_the_mirrors_for_dual_mirror_telescopes():
    (housing,) = pts.telescope_geometry(_dual_stub()).camera
    assert housing.points[:, 2].min() == pytest.approx(2.0888)
    assert housing.points[:, 2].max() == pytest.approx(2.5888)


def test_camera_without_body_diameter_is_not_drawn():
    stub = _segmented_stub()
    stub.parameters["camera_body_diameter"] = _record(0.0, "cm")
    assert pts.telescope_geometry(stub).camera == []


def test_mirror_offset_moves_all_elements():
    stub = _segmented_stub(mirror_offset=_record(10.0, "cm"))
    geometry = pts.telescope_geometry(stub)
    assert geometry.mirrors[0].points[:, 2].mean() == pytest.approx(-0.1)
    assert geometry.camera[0].points[0, 2] == pytest.approx(16.0 - 0.1)


def test_structure_table_cylinder_and_polyhedron(tmp_path):
    table = """# %ECSV 1.0
# ---
# datatype:
# - {name: id, datatype: string}
# - {name: group, datatype: string}
# - {name: shape, datatype: string}
# - {name: diameter, unit: m, datatype: float64}
# - {name: x1, unit: m, datatype: float64}
# - {name: y1, unit: m, datatype: float64}
# - {name: z1, unit: m, datatype: float64}
# - {name: x2, unit: m, datatype: float64}
# - {name: y2, unit: m, datatype: float64}
# - {name: z2, unit: m, datatype: float64}
# - {name: x3, unit: m, datatype: float64}
# - {name: y3, unit: m, datatype: float64}
# - {name: z3, unit: m, datatype: float64}
# meta: !!omap
# - {coordinate_frame: primary_mirror_frame}
id group shape diameter x1 y1 z1 x2 y2 z2 x3 y3 z3
mast-0 mast cylinder 0.1 1.0 0.0 0.5 1.0 0.0 10.0 nan nan nan
box-0 mast convex_polyhedron nan 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0
"""
    (tmp_path / "telescope_structure-1.0.0.ecsv").write_text(table, encoding="utf-8")
    stub = _segmented_stub(
        tmp_path,
        telescope_structure=_record("telescope_structure-1.0.0.ecsv"),
        mirror_offset=_record(10.0, "cm"),
    )
    elements = {e.name: e for e in pts.telescope_geometry(stub).structure}
    mast = elements["mast-0"]
    assert mast.is_cylinder
    assert mast.diameter == pytest.approx(0.1)
    assert mast.points[:, 2] == pytest.approx([0.4, 9.9])
    box = elements["box-0"]
    assert not box.is_cylinder
    assert box.points.shape == (3, 3)


def test_structure_in_another_frame_is_rejected(tmp_path):
    table = (
        "# %ECSV 1.0\n# ---\n# datatype:\n# - {name: id, datatype: string}\n"
        "# - {name: shape, datatype: string}\n# meta: !!omap\n# - {coordinate_frame: ground}\n"
        "id shape\nmast-0 cylinder\n"
    )
    (tmp_path / "telescope_structure-1.0.0.ecsv").write_text(table, encoding="utf-8")
    stub = _segmented_stub(tmp_path, telescope_structure=_record("telescope_structure-1.0.0.ecsv"))
    with pytest.raises(ValueError, match="coordinate_frame"):
        pts.telescope_geometry(stub)


def test_convex_hull_of_square_with_interior_point():
    ring = pts._convex_hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
    assert len(ring) == 5
    assert ring[0] == pytest.approx(ring[-1])


def test_plot_writes_figure(tmp_path):
    geometry = pts.telescope_geometry(_dual_stub())
    pts.plot_telescope_structure(geometry, tmp_path / "structure")
    assert list(tmp_path.glob("structure*.png"))


def test_plot_with_shares_writes_figure(tmp_path):
    geometry = pts.TelescopeGeometry(
        name="TEST",
        mirrors=pts.telescope_geometry(_segmented_stub()).mirrors,
        structure=[
            pts.Element("mast-0", "mast", np.array([[1.0, 0.0, 0.5], [1.0, 0.0, 10.0]]), 0.1),
            pts.Element("mast-1", "mast", np.array([[2.0, 0.0, 0.5], [2.0, 0.0, 10.0]]), 0.1),
        ],
    )
    shares = {"mast-0": (0.3, 100), "mast-1": (0.0, 0)}
    pts.plot_telescope_structure(geometry, tmp_path / "shares", shares)
    assert list(tmp_path.glob("shares*.png"))


def test_shadowing_without_obdeect_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "simtools.ray_tracing.backends.obdeect", None)
    with pytest.raises(ImportError, match="obdeect"):
        pts.shadowing_shares(SimpleNamespace(), 20.0, 10, 1)
