#!/usr/bin/python3

import logging

import astropy.units as u
import matplotlib.patches as mpatches
import numpy as np
import pytest

import simtools.util.general as gen
from simtools.layout.layout_array import LayoutArray

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def layout_center_data_dict():
    return {
        "center_lon": -17.8920302 * u.deg,
        "center_lat": 28.7621661 * u.deg,
        "center_easting": None,
        "center_northing": None,
        "EPSG": 32628,
        "center_alt": 2177 * u.m,
    }


@pytest.fixture
def corsika_telescope_data_dict():
    return {
        "corsika_sphere_radius": {"LST": 12.5 * u.m, "MST": 9.15 * u.m, "SST": 3.0 * u.m},
        "corsika_sphere_center": {"LST": 16 * u.m, "MST": 9 * u.m, "SST": 3.25 * u.m},
        "corsika_obs_level": 2158 * u.m,
    }


@pytest.fixture
def telescope_test_file(db, args_dict, io_handler):
    test_file_name = "telescope_positions-North-TestLayout.ecsv"
    db.export_file_db(
        db_name="test-data",
        dest=io_handler.get_output_directory(dir_type="model", test=True),
        file_name=test_file_name,
    )

    cfg_file = gen.find_file(
        test_file_name,
        io_handler.get_output_directory(dir_type="model", test=True),
    )
    return cfg_file


@pytest.fixture
def layout_array_instance():
    return LayoutArray(name="test_layout")


def test_from_layout_array_name(io_handler):

    layout = LayoutArray.from_layout_array_name("south-TestLayout")
    assert 99 == layout.get_number_of_telescopes()


def test_initialize_coordinate_systems(layout_center_data_dict, layout_array_instance):

    layout_array_instance._initialize_coordinate_systems()
    _x, _y, _z = layout_array_instance._array_center.get_coordinates("corsika")
    assert _x == 0.0 * u.m and _y == 0.0 * u.m and _z == 0.0 * u.m
    _lat, _lon, _z = layout_array_instance._array_center.get_coordinates("mercator")
    assert np.isnan(_lat) and np.isnan(_lon)

    layout_array_instance._initialize_coordinate_systems(layout_center_data_dict, False)
    _x, _y, _z = layout_array_instance._array_center.get_coordinates("corsika")
    assert _x == 0.0 * u.m and _y == 0.0 * u.m and _z == layout_center_data_dict["center_alt"]
    _lat, _lon, _z = layout_array_instance._array_center.get_coordinates("mercator")
    assert _lat.value == pytest.approx(layout_center_data_dict["center_lat"].value, 1.0e-2)
    assert _lon.value == pytest.approx(layout_center_data_dict["center_lon"].value, 1.0e-2)
    _E, _N, _z = layout_array_instance._array_center.get_coordinates("utm")
    assert _E.value == pytest.approx(217609.0, 1.0)
    assert _N.value == pytest.approx(3185067.0, 1.0)


def test_initialize_corsika_telescope_from_file(
    corsika_telescope_data_dict, args_dict, io_handler, layout_array_instance
):

    layout_array_instance._initialize_corsika_telescope()

    for key, value in corsika_telescope_data_dict["corsika_sphere_radius"].items():
        assert value == layout_array_instance._corsika_telescope["corsika_sphere_radius"][key]
    for key, value in corsika_telescope_data_dict["corsika_sphere_center"].items():
        assert value == layout_array_instance._corsika_telescope["corsika_sphere_center"][key]


def test_read_tel_list(telescope_test_file, layout_array_instance):

    layout_array_instance.read_telescope_list_file(telescope_test_file)
    layout_array_instance.convert_coordinates()
    assert 19 == layout_array_instance.get_number_of_telescopes()

    layout_2 = LayoutArray(name="test_layout", telescope_list_file=telescope_test_file)
    layout_2.convert_coordinates()
    assert 19 == layout_2.get_number_of_telescopes()


def test_add_tel(telescope_test_file, layout_array_instance):

    layout_array_instance.read_telescope_list_file(telescope_test_file)
    ntel_before = layout_array_instance.get_number_of_telescopes()
    layout_array_instance.add_telescope("LST-05", "corsika", 100.0 * u.m, 50.0 * u.m, 2177.0 * u.m)
    ntel_after = layout_array_instance.get_number_of_telescopes()
    assert ntel_before + 1 == ntel_after

    layout_array_instance.add_telescope(
        "LST-05", "corsika", 100.0 * u.m, 50.0 * u.m, None, 50.0 * u.m
    )
    assert layout_array_instance._telescope_list[-1].get_altitude().value == pytest.approx(2192.0)


def test_build_layout(
    layout_center_data_dict,
    corsika_telescope_data_dict,
    tmp_test_directory,
    io_handler,
    layout_array_instance,
):

    layout = LayoutArray(
        label="test_layout",
        name="LST4",
        layout_center_data=layout_center_data_dict,
        corsika_telescope_data=corsika_telescope_data_dict,
    )

    layout.add_telescope(
        telescope_name="LST-01",
        crs_name="corsika",
        xx=57.5 * u.m,
        yy=57.5 * u.m,
        tel_corsika_z=0 * u.m,
    )
    layout.add_telescope(
        telescope_name="LST-02",
        crs_name="corsika",
        xx=-57.5 * u.m,
        yy=57.5 * u.m,
        tel_corsika_z=0 * u.m,
    )
    layout.add_telescope(
        telescope_name="LST-02",
        crs_name="corsika",
        xx=57.5 * u.m,
        yy=-57.5 * u.m,
        tel_corsika_z=0 * u.m,
    )
    layout.add_telescope(
        telescope_name="LST-04",
        crs_name="corsika",
        xx=-57.5 * u.m,
        yy=-57.5 * u.m,
        tel_corsika_z=0 * u.m,
    )

    layout.convert_coordinates()
    layout.print_telescope_list()
    layout.export_telescope_list(crs_name="corsika")

    # Building a second layout from the file exported by the first one
    layout_array_instance.read_telescope_list_file(layout.telescope_list_file)

    assert 4 == layout_array_instance.get_number_of_telescopes()
    assert layout_array_instance._array_center.get_altitude().value == pytest.approx(
        layout._array_center.get_altitude().value, 1.0e-2
    )


def test_converting_center_coordinates(layout_center_data_dict, corsika_telescope_data_dict):

    layout = LayoutArray(
        label="test_layout",
        name="LST4",
        layout_center_data=layout_center_data_dict,
        corsika_telescope_data=corsika_telescope_data_dict,
    )

    _lat, _lon, _ = layout._array_center.get_coordinates("mercator")
    assert _lat.value == pytest.approx(28.7621661)
    assert _lon.value == pytest.approx(-17.8920302)

    _east, _north, _ = layout._array_center.get_coordinates("utm")
    assert _north.value == pytest.approx(3185067.28)
    assert _east.value == pytest.approx(217609.23)

    assert layout._array_center.get_altitude().value == pytest.approx(2177.0)


def test_get_corsika_input_list(
    layout_center_data_dict, corsika_telescope_data_dict, telescope_test_file
):

    layout = LayoutArray(
        label="test_layout",
        name="LST4",
        layout_center_data=layout_center_data_dict,
        corsika_telescope_data=corsika_telescope_data_dict,
    )
    layout.add_telescope(
        telescope_name="LST-01",
        crs_name="corsika",
        xx=57.5 * u.m,
        yy=57.5 * u.m,
        tel_corsika_z=0 * u.m,
    )
    corsika_input_list = layout.get_corsika_input_list()

    assert corsika_input_list == "TELESCOPE\t 57.500E2\t 57.500E2\t 0.000E2\t 12.500E2\t # LST-01\n"


def test_altitude_from_corsika_z(layout_center_data_dict, corsika_telescope_data_dict):

    layout = LayoutArray(
        label="test_layout",
        name="LST4",
        layout_center_data=layout_center_data_dict,
        corsika_telescope_data=corsika_telescope_data_dict,
    )

    layout.add_telescope(
        telescope_name="LST-01",
        crs_name="corsika",
        xx=57.5 * u.m,
        yy=57.5 * u.m,
        tel_corsika_z=0 * u.m,
    )

    assert layout._altitude_from_corsika_z(5.0 * u.m, None, "LST-01").value == pytest.approx(2147.0)
    assert layout._altitude_from_corsika_z(None, 8848.0 * u.m, "LST-01").value == pytest.approx(
        6706.0
    )
    with pytest.raises(TypeError):
        layout._altitude_from_corsika_z(5.0, None, "LST-01")


def test_telescope_layout_file_to_dict(telescope_test_file):

    layout = LayoutArray(name="test_layout", telescope_list_file=telescope_test_file)
    telescopes_dict = layout.telescope_layout_file_to_dict(telescope_test_file)

    values_from_file = [20.190000534057617, -352.4599914550781, 62.29999923706055, 9.6]
    keys = ["pos_x", "pos_y", "pos_z", "radius"]
    MST10_index = telescopes_dict["telescope_name"] == "MST-10"
    for key_step in range(len(keys)):
        assert telescopes_dict[MST10_index][keys[key_step]].value[0] == values_from_file[key_step]


def test_get_telescope_patch(corsika_telescope_data_dict, io_handler, layout_array_instance):
    for tel_type in np.array(list(corsika_telescope_data_dict["corsika_sphere_radius"].keys())):
        radius = corsika_telescope_data_dict["corsika_sphere_radius"][tel_type]
        patch = layout_array_instance._get_telescope_patch(
            tel_type, 0 * u.m, 0 * u.m, radius.value * u.m
        )
        assert mpatches.Circle == type(patch)


# def test_rotate_telescope_position(x, y, rotate_angle_deg):
# layout = LayoutArray(name="test_layout")
