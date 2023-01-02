#!/usr/bin/python3

import logging

import astropy.units as u
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
def layout_array_north_four_LST_instance(
    layout_center_data_dict, corsika_telescope_data_dict, io_handler, db_config
):
    layout = LayoutArray(
        site="North",
        mongo_db_config=db_config,
        label="test_layout",
        name="LST4",
        layout_center_data=layout_center_data_dict,
        corsika_telescope_data=corsika_telescope_data_dict,
    )
    return layout


@pytest.fixture
def manual_corsika_dict_north():
    return {
        "corsika_sphere_radius": {
            "LST": 12.5 * u.m,
            "MST": 9.15 * u.m,
            "SCT": 7.15 * u.m,
            "SST": 3 * u.m,
        },
        "corsika_sphere_center": {
            "LST": 16 * u.m,
            "MST": 9 * u.m,
            "SCT": 6.1 * u.m,
            "SST": 3.25 * u.m,
        },
        "corsika_obs_level": 2158 * u.m,
    }


def test_from_layout_array_name(io_handler, db_config):

    layout = LayoutArray.from_layout_array_name(
        mongo_db_config=db_config, layout_array_name="south-TestLayout"
    )

    assert 99 == layout.get_number_of_telescopes()


def test_initialize_coordinate_systems(layout_center_data_dict, layout_array_north_instance):

    layout_array_north_instance._initialize_coordinate_systems()
    _x, _y, _z = layout_array_north_instance._array_center.get_coordinates("corsika")
    assert _x == 0.0 * u.m and _y == 0.0 * u.m and _z == 0.0 * u.m
    _lat, _lon, _z = layout_array_north_instance._array_center.get_coordinates("mercator")
    assert np.isnan(_lat) and np.isnan(_lon)

    layout_array_north_instance._initialize_coordinate_systems(layout_center_data_dict)
    _x, _y, _z = layout_array_north_instance._array_center.get_coordinates("corsika")
    assert _x == 0.0 * u.m and _y == 0.0 * u.m and _z == layout_center_data_dict["center_alt"]
    _lat, _lon, _z = layout_array_north_instance._array_center.get_coordinates("mercator")
    assert _lat.value == pytest.approx(layout_center_data_dict["center_lat"].value, 1.0e-2)
    assert _lon.value == pytest.approx(layout_center_data_dict["center_lon"].value, 1.0e-2)
    _E, _N, _z = layout_array_north_instance._array_center.get_coordinates("utm")
    assert _E.value == pytest.approx(217609.0, 1.0)
    assert _N.value == pytest.approx(3185067.0, 1.0)


def test_initialize_corsika_telescope_from_file(
    corsika_telescope_data_dict, args_dict, io_handler, layout_array_north_instance
):

    layout_array_north_instance._initialize_corsika_telescope()

    for key, value in corsika_telescope_data_dict["corsika_sphere_radius"].items():
        assert value == layout_array_north_instance._corsika_telescope["corsika_sphere_radius"][key]
    for key, value in corsika_telescope_data_dict["corsika_sphere_center"].items():
        assert value == layout_array_north_instance._corsika_telescope["corsika_sphere_center"][key]


def test_read_telescope_list_file(telescope_test_file, layout_array_north_instance, db_config):
    table = layout_array_north_instance.read_telescope_list_file(telescope_test_file)
    assert table.meta["data_type"] == "positionfile"
    assert table.meta["description"] == "telescope positions for CTA North (La Palma)"
    columns = ["pos_x", "pos_y", "pos_z"]
    pos_x = [-70.93, -35.27, 75.28, 30.91, -211.54, -153.26]
    pos_y = [-52.07, 66.14, 50.49, -64.54, 5.66, 169.01]
    pos_z = [43.00, 32.00, 28.70, 32.00, 50.3, 24.0]
    columns_manual = [pos_x, pos_y, pos_z]
    for step, col in enumerate(columns):
        assert all(
            [
                pytest.approx(table[col][tel_num].value, 0.1) == columns_manual[step][tel_num]
                for tel_num in range(6)
            ]
        )


def test_initialize_layout_array_from_telescope_file(
    telescope_test_file, layout_array_north_instance, db_config
):

    layout_array_north_instance.initialize_layout_array_from_telescope_file(telescope_test_file)
    layout_array_north_instance.convert_coordinates()
    assert 19 == layout_array_north_instance.get_number_of_telescopes()

    layout_2 = LayoutArray(
        name="test_layout",
        telescope_list_file=telescope_test_file,
    )
    layout_2.convert_coordinates()
    assert 19 == layout_2.get_number_of_telescopes()


def test_add_tel(telescope_test_file, layout_array_north_instance):

    layout_array_north_instance.read_telescope_list_file(telescope_test_file)
    ntel_before = layout_array_north_instance.get_number_of_telescopes()
    layout_array_north_instance.add_telescope(
        "LST-05", "corsika", 100.0 * u.m, 50.0 * u.m, 2177.0 * u.m
    )
    ntel_after = layout_array_north_instance.get_number_of_telescopes()
    assert ntel_before + 1 == ntel_after

    layout_array_north_instance.add_telescope(
        "LST-05", "corsika", 100.0 * u.m, 50.0 * u.m, None, 50.0 * u.m
    )
    assert layout_array_north_instance._telescope_list[-1].get_altitude().value == pytest.approx(
        2192.0
    )


def test_build_layout(layout_array_north_four_LST_instance, tmp_test_directory, db_config):

    layout = layout_array_north_four_LST_instance

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
    layout_2 = LayoutArray(site=layout.site, mongo_db_config=db_config, name="test_layout_2")
    layout_2.initialize_layout_array_from_telescope_file(layout.telescope_list_file)

    assert 4 == layout_2.get_number_of_telescopes()
    assert layout_2._array_center.get_altitude().value == pytest.approx(
        layout._array_center.get_altitude().value, 1.0e-2
    )


def test_converting_center_coordinates(layout_array_north_four_LST_instance):

    layout = layout_array_north_four_LST_instance

    _lat, _lon, _ = layout._array_center.get_coordinates("mercator")
    assert _lat.value == pytest.approx(28.7621661)
    assert _lon.value == pytest.approx(-17.8920302)

    _east, _north, _ = layout._array_center.get_coordinates("utm")
    assert _north.value == pytest.approx(3185067.28)
    assert _east.value == pytest.approx(217609.23)

    assert layout._array_center.get_altitude().value == pytest.approx(2177.0)


def test_get_corsika_input_list(layout_array_north_four_LST_instance):

    layout = layout_array_north_four_LST_instance
    layout.add_telescope(
        telescope_name="LST-01",
        crs_name="corsika",
        xx=57.5 * u.m,
        yy=57.5 * u.m,
        tel_corsika_z=0 * u.m,
    )
    corsika_input_list = layout.get_corsika_input_list()

    assert corsika_input_list == "TELESCOPE\t 57.500E2\t 57.500E2\t 0.000E2\t 12.500E2\t # LST-01\n"


def test_altitude_from_corsika_z(layout_array_north_four_LST_instance):

    layout = layout_array_north_four_LST_instance

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


def test_include_radius_into_telescope_table(layout_array_north_instance, telescope_test_file):
    telescope_table = layout_array_north_instance.read_telescope_list_file(telescope_test_file)
    telescope_table_with_radius = layout_array_north_instance.include_radius_into_telescope_table(
        telescope_table
    )
    values_from_file = [20.190000534057617, -352.4599914550781, 62.29999923706055, 9.6]
    keys = ["pos_x", "pos_y", "pos_z", "radius"]
    mst_10_index = telescope_table_with_radius["telescope_name"] == "MST-10"
    for key, value_manual in zip(keys, values_from_file):
        assert (
            pytest.approx(telescope_table_with_radius[mst_10_index][key].value[0], 1e-2)
            == value_manual
        )


def test_from_corsika_file_to_dict(
    layout_array_north_instance, manual_corsika_dict_north, db, io_handler
):
    def run(corsika_dict):
        for key, value in corsika_dict.items():
            if isinstance(value, dict):
                for tel_type, subvalue in value.items():
                    assert subvalue == manual_corsika_dict_north[key][tel_type]
            else:
                assert value == manual_corsika_dict_north[key]

    corsika_dict = layout_array_north_instance._from_corsika_file_to_dict()
    run(corsika_dict)

    test_file_name = "corsika_parameters_2.yml"
    db.export_file_db(
        db_name="test-data",
        dest=io_handler.get_output_directory(dir_type="parameters", test=True),
        file_name=test_file_name,
    )

    corsika_config_file = gen.find_file(
        test_file_name, io_handler.get_output_directory(dir_type="parameters", test=True)
    )
    corsika_dict = layout_array_north_instance._from_corsika_file_to_dict(
        file_name=corsika_config_file
    )
    run(corsika_dict)

    with pytest.raises(FileNotFoundError):
        corsika_dict = layout_array_north_instance._from_corsika_file_to_dict(
            file_name="file_doesnt_exist.yml"
        )


def test_initialize_corsika_telescope_from_dict(
    layout_array_north_instance, manual_corsika_dict_north
):
    layout_array_north_instance._initialize_corsika_telescope_from_dict(manual_corsika_dict_north)
    for key, value in layout_array_north_instance._corsika_telescope.items():
        if isinstance(value, dict):
            for tel_type, subvalue in value.items():
                assert subvalue == manual_corsika_dict_north[key][tel_type]
        else:
            assert value == manual_corsika_dict_north[key]


def test_assign_unit_to_quantity(layout_array_north_instance):
    quantity = layout_array_north_instance._assign_unit_to_quantity(10, u.m)
    assert quantity == 10 * u.m

    quantity = layout_array_north_instance._assign_unit_to_quantity(1000 * u.cm, u.m)
    assert quantity == 10 * u.m

    with pytest.raises(u.UnitConversionError):
        layout_array_north_instance._assign_unit_to_quantity(1000 * u.TeV, u.m)


def test_try_set_altitude(layout_array_north_instance, telescope_test_file):
    table = layout_array_north_instance.read_telescope_list_file(telescope_test_file)
    obs_level = 2158.0
    manual_z_positions = [43.00, 32.00, 28.70, 32.00, 50.3, 24.0]
    corsika_sphere_center = [16.0, 16.0, 16.0, 16.0, 9.0, 9.0]
    manual_altitudes = [
        manual_z_positions[step] + obs_level - corsika_sphere_center[step] for step in range(6)
    ]
    for step, row in enumerate(table[:6]):
        tel = layout_array_north_instance._load_telescope_names(row)
        layout_array_north_instance._try_set_altitude(row, tel, table)
        for _crs in tel.crs.values():
            assert pytest.approx(_crs["zz"]["value"], 0.1) == manual_altitudes[step]


def test_try_set_coordinate(layout_array_north_instance, telescope_test_file):
    table = layout_array_north_instance.read_telescope_list_file(telescope_test_file)
    manual_xx = [-70.93, -35.27, 75.28, 30.91, -211.54, -153.26]
    manual_yy = [-52.07, 66.14, 50.49, -64.54, 5.66, 169.01]
    for step, row in enumerate(table[:6]):
        tel = layout_array_north_instance._load_telescope_names(row)
        layout_array_north_instance._try_set_coordinate(
            row, tel, table, "corsika", "pos_x", "pos_y"
        )
        assert pytest.approx(tel.crs["corsika"]["xx"]["value"], 0.1) == manual_xx[step]
        assert pytest.approx(tel.crs["corsika"]["yy"]["value"], 0.1) == manual_yy[step]
