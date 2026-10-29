"""Tests for deriving telescope geometry from model parameters."""

from types import SimpleNamespace

import astropy.units as u
from astropy.table import Table

from simtools.model import telescope_geometry


def test_telescope_geometry_builds_segmented_mirrors():
    parameters = {
        "focal_length": {"value": 1600.0, "unit": "cm"},
        "dish_shape_length": {"value": 1920.0, "unit": "cm"},
        "camera_body_diameter": {"value": 289.0, "unit": "cm"},
        "camera_body_shape": {"value": 2, "unit": None},
        "camera_depth": {"value": 0.0, "unit": "cm"},
    }
    mirror_table = Table(
        {
            "mirror_x": u.Quantity([0.0], u.cm),
            "mirror_y": u.Quantity([0.0], u.cm),
            "mirror_z": u.Quantity([0.0], u.cm),
            "mirror_diameter": u.Quantity([120.0], u.cm),
            "shape_type": [1.0],
        }
    )
    model = SimpleNamespace(
        name="TEST",
        parameters=parameters,
        config_file_directory=None,
        mirrors=SimpleNamespace(mirror_table=mirror_table),
    )

    geometry = telescope_geometry.telescope_geometry(model)

    assert geometry.name == "TEST"
    assert len(geometry.mirrors) == 1
    assert geometry.camera[0].name == "camera-housing"
