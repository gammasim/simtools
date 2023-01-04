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
def north_layout_center_data_dict():
    return {
        "center_lon": -17.8920302 * u.deg,
        "center_lat": 28.7621661 * u.deg,
        "center_easting": 217611 * u.m,
        "center_northing": 3185066 * u.m,
        "EPSG": 32628,
        "center_alt": 2177 * u.m,
    }


@pytest.fixture
def south_layout_center_data_dict():
    return {
        "center_lon": -70.316345 * u.deg,
        "center_lat": -24.683429 * u.deg,
        "center_easting": 366822 * u.m,
        "center_northing": 7269466 * u.m,
        "EPSG": 32628,
        "center_alt": 2162.35 * u.m,
    }


@pytest.fixture
def layout_array_north_four_LST_instance(
    north_layout_center_data_dict, manual_corsika_dict_north, io_handler, db_config
):
    layout = LayoutArray(
        site="North",
        mongo_db_config=db_config,
        label="test_layout",
        name="LST4",
        layout_center_data=north_layout_center_data_dict,
        corsika_telescope_data=manual_corsika_dict_north,
    )
    return layout


@pytest.fixture
def layout_array_south_four_LST_instance(
    south_layout_center_data_dict, manual_corsika_dict_south, io_handler, db_config
):
    layout = LayoutArray(
        site="South",
        mongo_db_config=db_config,
        label="test_layout",
        name="LST4",
        layout_center_data=south_layout_center_data_dict,
        corsika_telescope_data=manual_corsika_dict_south,
    )
    return layout


def test_from_layout_array_name(io_handler, db_config):

    layout = LayoutArray.from_layout_array_name(
        mongo_db_config=db_config, layout_array_name="south-TestLayout"
    )
    assert 99 == layout.get_number_of_telescopes()
    layout = LayoutArray.from_layout_array_name(
        mongo_db_config=db_config, layout_array_name="north-TestLayout"
    )
    assert 19 == layout.get_number_of_telescopes()


def test_initialize_coordinate_systems(
    north_layout_center_data_dict,
    layout_array_north_instance,
    south_layout_center_data_dict,
    layout_array_south_instance,
):
    def test_one_site(center_data_dict, instance, easting, northing):
        instance._initialize_coordinate_systems()
        _x, _y, _z = instance._array_center.get_coordinates("corsika")
        assert _x == 0.0 * u.m and _y == 0.0 * u.m and _z == 0.0 * u.m
        _lat, _lon, _z = instance._array_center.get_coordinates("mercator")
        assert np.isnan(_lat) and np.isnan(_lon)

        instance._initialize_coordinate_systems(center_data_dict)
        _x, _y, _z = instance._array_center.get_coordinates("corsika")
        assert _x == 0.0 * u.m and _y == 0.0 * u.m and _z == center_data_dict["center_alt"]
        _lat, _lon, _z = instance._array_center.get_coordinates("mercator")
        assert _lat.value == pytest.approx(center_data_dict["center_lat"].value, 1.0e-2)
        assert _lon.value == pytest.approx(center_data_dict["center_lon"].value, 1.0e-2)

        _E, _N, _z = instance._array_center.get_coordinates("utm")
        assert _E.value == pytest.approx(easting, 1.0)
        assert _N.value == pytest.approx(northing, 1.0)

    test_one_site(north_layout_center_data_dict, layout_array_north_instance, 217611.0, 3185066.0)
    test_one_site(south_layout_center_data_dict, layout_array_south_instance, 366822.0, 7269466.0)


def test_initialize_corsika_telescope_from_file(
    manual_corsika_dict_north,
    layout_array_north_instance,
    manual_corsika_dict_south,
    layout_array_south_instance,
    args_dict,
    io_handler,
):
    def test_one_site(instance, corsika_dict):
        layout_array_north_instance._initialize_corsika_telescope()

        for key, value in corsika_dict["corsika_sphere_radius"].items():
            assert value == instance._corsika_telescope["corsika_sphere_radius"][key]
        for key, value in corsika_dict["corsika_sphere_center"].items():
            assert value == instance._corsika_telescope["corsika_sphere_center"][key]

    test_one_site(layout_array_north_instance, manual_corsika_dict_north)
    test_one_site(layout_array_south_instance, manual_corsika_dict_south)


def test_read_telescope_list_file(telescope_north_test_file, telescope_south_test_file, db_config):

    pos_x_north = [-70.93, -35.27, 75.28, 30.91, -211.54, -153.26]
    pos_y_north = [-52.07, 66.14, 50.49, -64.54, 5.66, 169.01]
    pos_z_north = [43.00, 32.00, 28.70, 32.00, 50.3, 24.0]
    description_north = "telescope positions for CTA North (La Palma)"
    pos_x_south = [-20.643, 79.994, -19.396, -120.033, -0.017, -1.468]
    pos_y_south = [-64.817, -0.768, 65.200, 1.151, -0.001, -151.221]
    pos_z_south = [34.30, 29.40, 31.00, 33.10, 24.35, 31.00]
    description_south = "telescope positions for CTA South (Paranal)"

    def test_one_site(test_file, pos_x, pos_y, pos_z, description):
        table = LayoutArray.read_telescope_list_file(test_file)
        assert table.meta["data_type"] == "positionfile"
        assert table.meta["description"] == description
        columns = ["pos_x", "pos_y", "pos_z"]

        columns_manual = [pos_x, pos_y, pos_z]
        for step, col in enumerate(columns):
            assert all(
                [
                    pytest.approx(table[col][tel_num].value, 0.1) == columns_manual[step][tel_num]
                    for tel_num in range(6)
                ]
            )

    test_one_site(
        telescope_north_test_file, pos_x_north, pos_y_north, pos_z_north, description_north
    )
    test_one_site(
        telescope_south_test_file, pos_x_south, pos_y_south, pos_z_south, description_south
    )


def test_initialize_layout_array_from_telescope_file(
    telescope_north_test_file,
    layout_array_north_instance,
    telescope_south_test_file,
    layout_array_south_instance,
    db_config,
):
    def test_one_site(instance, test_file, number_of_telescopes, label):
        instance.initialize_layout_array_from_telescope_file(test_file)
        instance.convert_coordinates()
        assert number_of_telescopes == instance.get_number_of_telescopes()

        layout_2 = LayoutArray(
            site=label,
            mongo_db_config=db_config,
            name="test_layout",
            telescope_list_file=test_file,
        )
        layout_2.convert_coordinates()
        assert number_of_telescopes == layout_2.get_number_of_telescopes()

    test_one_site(layout_array_north_instance, telescope_north_test_file, 19, "North")
    test_one_site(layout_array_south_instance, telescope_south_test_file, 99, "South")


def test_add_tel(
    telescope_north_test_file,
    layout_array_north_instance,
    telescope_south_test_file,
    layout_array_south_instance,
):
    def test_one_site(instance, test_file, altitude):
        LayoutArray.read_telescope_list_file(test_file)
        ntel_before = instance.get_number_of_telescopes()
        instance.add_telescope("LST-00", "corsika", 100.0 * u.m, 50.0 * u.m, 2177.0 * u.m)
        ntel_after = instance.get_number_of_telescopes()
        assert ntel_before + 1 == ntel_after

        instance.add_telescope("LST-00", "corsika", 100.0 * u.m, 50.0 * u.m, None, 50.0 * u.m)
        assert instance._telescope_list[-1].get_altitude().value == pytest.approx(altitude)

    test_one_site(layout_array_north_instance, telescope_north_test_file, 2192.0)
    test_one_site(layout_array_south_instance, telescope_south_test_file, 2181.0)


def test_build_layout(
    layout_array_north_four_LST_instance,
    layout_array_south_four_LST_instance,
    tmp_test_directory,
    db_config,
):
    def test_one_site(layout, site_label):
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
        layout_2 = LayoutArray(site=site_label, mongo_db_config=db_config, name="test_layout_2")
        layout_2.initialize_layout_array_from_telescope_file(layout.telescope_list_file)

        assert 4 == layout_2.get_number_of_telescopes()
        assert layout_2._array_center.get_altitude().value == pytest.approx(
            layout._array_center.get_altitude().value, 1.0e-2
        )

    test_one_site(layout_array_north_four_LST_instance, "North")
    test_one_site(layout_array_south_four_LST_instance, "South")


def test_converting_center_coordinates_north(layout_array_north_four_LST_instance):

    layout = layout_array_north_four_LST_instance

    _lat, _lon, _ = layout._array_center.get_coordinates("mercator")
    assert _lat.value == pytest.approx(28.7621661)
    assert _lon.value == pytest.approx(-17.8920302)

    _east, _north, _ = layout._array_center.get_coordinates("utm")
    assert _north.value == pytest.approx(3185066.0)
    assert _east.value == pytest.approx(217611.0)

    assert layout._array_center.get_altitude().value == pytest.approx(2177.0)


def test_converting_center_coordinates_south(layout_array_south_four_LST_instance):

    layout = layout_array_south_four_LST_instance

    _lat, _lon, _ = layout._array_center.get_coordinates("mercator")
    assert _lat.value == pytest.approx(-24.68342915473787)
    assert _lon.value == pytest.approx(-70.31634499364885)

    _east, _north, _ = layout._array_center.get_coordinates("utm")
    assert _north.value == pytest.approx(7269466.0)
    assert _east.value == pytest.approx(366822.0)

    assert layout._array_center.get_altitude().value == pytest.approx(2162.35)


def test_get_corsika_input_list(
    layout_array_north_four_LST_instance, layout_array_south_four_LST_instance
):
    def test_one_site(layout):

        layout.add_telescope(
            telescope_name="LST-01",
            crs_name="corsika",
            xx=57.5 * u.m,
            yy=57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )
        corsika_input_list = layout.get_corsika_input_list()

        assert (
            corsika_input_list
            == "TELESCOPE\t 57.500E2\t 57.500E2\t 0.000E2\t 12.500E2\t # LST-01\n"
        )

    test_one_site(layout_array_north_four_LST_instance)
    test_one_site(layout_array_south_four_LST_instance)


def test_altitude_from_corsika_z(
    layout_array_north_four_LST_instance, layout_array_south_four_LST_instance
):
    def test_one_site(instance, result1, result2):

        instance.add_telescope(
            telescope_name="LST-01",
            crs_name="corsika",
            xx=57.5 * u.m,
            yy=57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )

        assert instance._altitude_from_corsika_z(
            pos_z=5.0 * u.m, altitude=None, tel_name="LST-01"
        ).value == pytest.approx(result1)
        assert instance._altitude_from_corsika_z(
            pos_z=None, altitude=8848.0 * u.m, tel_name="LST-01"
        ).value == pytest.approx(result2)
        with pytest.raises(TypeError):
            instance._altitude_from_corsika_z(5.0, None, "LST-01")

    test_one_site(layout_array_north_four_LST_instance, 2147.0, 6706.0)
    test_one_site(layout_array_south_four_LST_instance, 2136.0, 6717.0)


def test_include_radius_into_telescope_table(telescope_north_test_file, telescope_south_test_file):

    values_from_file_north = [20.190000534057617, -352.4599914550781, 62.29999923706055, 9.6]
    values_from_file_south = [-151.949, 240.011, 27.00]

    def test_one_site(test_file, values_from_file):
        telescope_table = LayoutArray.read_telescope_list_file(test_file)
        telescope_table_with_radius = LayoutArray.include_radius_into_telescope_table(
            telescope_table
        )

        keys = ["pos_x", "pos_y", "pos_z", "radius"]
        mst_10_index = telescope_table_with_radius["telescope_name"] == "MST-10"
        for key, value_manual in zip(keys, values_from_file):
            assert (
                pytest.approx(telescope_table_with_radius[mst_10_index][key].value[0], 1e-2)
                == value_manual
            )

    test_one_site(telescope_north_test_file, values_from_file_north)
    test_one_site(telescope_south_test_file, values_from_file_south)


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
    layout_array_north_instance,
    manual_corsika_dict_north,
    layout_array_south_instance,
    manual_corsika_dict_south,
):
    def test_one_site(instance, corsika_dict):
        instance._initialize_corsika_telescope_from_dict(corsika_dict)
        for key, value in instance._corsika_telescope.items():
            if isinstance(value, dict):
                for tel_type, subvalue in value.items():
                    assert subvalue == corsika_dict[key][tel_type]
            else:
                assert value == corsika_dict[key]

    test_one_site(layout_array_north_instance, manual_corsika_dict_north)
    test_one_site(layout_array_south_instance, manual_corsika_dict_south)


def test_assign_unit_to_quantity(layout_array_north_instance):
    quantity = layout_array_north_instance._assign_unit_to_quantity(10, u.m)
    assert quantity == 10 * u.m

    quantity = layout_array_north_instance._assign_unit_to_quantity(1000 * u.cm, u.m)
    assert quantity == 10 * u.m

    with pytest.raises(u.UnitConversionError):
        layout_array_north_instance._assign_unit_to_quantity(1000 * u.TeV, u.m)


def test_try_set_altitude(
    layout_array_north_instance,
    telescope_north_test_file,
    layout_array_south_instance,
    telescope_south_test_file,
):
    obs_level_north = 2158.0
    manual_z_positions_north = [43.00, 32.00, 28.70, 32.00, 50.3, 24.0]
    corsika_sphere_center_north = [16.0, 16.0, 16.0, 16.0, 9.0, 9.0]

    obs_level_south = 2147.0
    manual_z_positions_south = [34.30, 29.40, 31.00, 33.10, 24.35, 31.00]
    corsika_sphere_center_south = [16.0, 16.0, 16.0, 16.0, 9.0, 9.0]

    def test_one_site(test_file, instance, obs_level, manual_z_positions, corsika_sphere_center):
        table = LayoutArray.read_telescope_list_file(test_file)
        manual_altitudes = [
            manual_z_positions[step] + obs_level - corsika_sphere_center[step] for step in range(6)
        ]
        for step, row in enumerate(table[:6]):
            tel = instance._load_telescope_names(row)
            instance._try_set_altitude(row, tel, table)
            for _crs in tel.crs.values():
                assert pytest.approx(_crs["zz"]["value"], 0.01) == manual_altitudes[step]

    test_one_site(
        telescope_north_test_file,
        layout_array_north_instance,
        obs_level_north,
        manual_z_positions_north,
        corsika_sphere_center_north,
    )
    test_one_site(
        telescope_south_test_file,
        layout_array_south_instance,
        obs_level_south,
        manual_z_positions_south,
        corsika_sphere_center_south,
    )


def test_try_set_coordinate(
    layout_array_north_instance,
    telescope_north_test_file,
    layout_array_south_instance,
    telescope_south_test_file,
):
    manual_xx_north = [-70.93, -35.27, 75.28, 30.91, -211.54, -153.26]
    manual_yy_north = [-52.07, 66.14, 50.49, -64.54, 5.66, 169.01]

    manual_xx_south = [-20.643, 79.994, -19.396, -120.033, -0.017, -1.468]
    manual_yy_south = [-64.817, -0.768, 65.200, 1.151, -0.001, -151.221]

    def test_one_site(instance, test_file, manual_xx, manual_yy):
        table = LayoutArray.read_telescope_list_file(test_file)
        for step, row in enumerate(table[:6]):
            tel = instance._load_telescope_names(row)
            instance._try_set_coordinate(row, tel, table, "corsika", "pos_x", "pos_y")
            assert pytest.approx(tel.crs["corsika"]["xx"]["value"], 0.1) == manual_xx[step]
            assert pytest.approx(tel.crs["corsika"]["yy"]["value"], 0.1) == manual_yy[step]

    test_one_site(
        layout_array_north_instance, telescope_north_test_file, manual_xx_north, manual_yy_north
    )
    test_one_site(
        layout_array_south_instance, telescope_south_test_file, manual_xx_south, manual_yy_south
    )
