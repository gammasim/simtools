#!/usr/bin/python3

import logging

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

import simtools.utils.general as gen
from simtools.data_model import data_reader
from simtools.layout.array_layout import ArrayLayout, InvalidTelescopeListFile

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def north_layout_center_data_dict():
    return {
        "center_lon": -17.8920302 * u.deg,
        "center_lat": 28.7621661 * u.deg,
        "center_easting": 217611.227 * u.m,
        "center_northing": 3185066.278 * u.m,
        "EPSG": 32628,
        "center_alt": 2177 * u.m,
    }


@pytest.fixture
def south_layout_center_data_dict():
    return {
        "center_lon": -70.316345 * u.deg,
        "center_lat": -24.683429 * u.deg,
        "center_easting": 366822.017 * u.m,
        "center_northing": 7269466.999 * u.m,
        "EPSG": 32719,
        "center_alt": 2162.35 * u.m,
    }


@pytest.fixture
def array_layout_north_four_LST_instance(
    north_layout_center_data_dict, manual_corsika_dict_north, db_config
):
    layout = ArrayLayout(
        site="North",
        mongo_db_config=db_config,
        label="test_layout",
        name="LST4",
        layout_center_data=north_layout_center_data_dict,
        corsika_telescope_data=manual_corsika_dict_north,
    )
    return layout


@pytest.fixture
def array_layout_south_four_LST_instance(
    south_layout_center_data_dict, manual_corsika_dict_south, db_config
):
    layout = ArrayLayout(
        site="South",
        mongo_db_config=db_config,
        label="test_layout",
        name="LST4",
        layout_center_data=south_layout_center_data_dict,
        corsika_telescope_data=manual_corsika_dict_south,
    )
    return layout


def test_array_layout_empty():
    layout = ArrayLayout()
    assert layout.get_number_of_telescopes() == 0


def test_from_array_layout_name(io_handler, db_config):
    layout = ArrayLayout.from_array_layout_name(
        mongo_db_config=db_config, array_layout_name="South-4LST"
    )
    assert 4 == layout.get_number_of_telescopes()
    layout = ArrayLayout.from_array_layout_name(
        mongo_db_config=db_config, array_layout_name="North-4MST"
    )
    assert 4 == layout.get_number_of_telescopes()


def test_initialize_coordinate_systems(
    north_layout_center_data_dict,
    array_layout_north_instance,
    south_layout_center_data_dict,
    array_layout_south_instance,
):
    def test_one_site(center_data_dict, instance, easting, northing):
        instance._initialize_coordinate_systems()
        _x, _y, _z = instance._array_center.get_coordinates("ground")
        assert _x == 0.0 * u.m and _y == 0.0 * u.m
        assert np.isnan(_z.value)
        _lat, _lon, _z = instance._array_center.get_coordinates("mercator")
        assert np.isnan(_lat) and np.isnan(_lon)

        if center_data_dict is not None:
            instance._initialize_coordinate_systems(center_data_dict)
            _x, _y, _z = instance._array_center.get_coordinates("ground")
            assert _x == 0.0 * u.m and _y == 0.0 * u.m and _z == center_data_dict["center_alt"]
            _lat, _lon, _z = instance._array_center.get_coordinates("mercator")
            assert _lat.value == pytest.approx(center_data_dict["center_lat"].value, 1.0e-2)
            assert _lon.value == pytest.approx(center_data_dict["center_lon"].value, 1.0e-2)

            _E, _N, _z = instance._array_center.get_coordinates("utm")
            assert _E.value == pytest.approx(easting, 1.0)
            assert _N.value == pytest.approx(northing, 1.0)

    test_one_site(
        north_layout_center_data_dict, array_layout_north_instance, 217611.227, 3185066.278
    )
    test_one_site(
        south_layout_center_data_dict, array_layout_south_instance, 366822.017, 7269466.999
    )
    test_one_site(None, array_layout_south_instance, 366822.017, 7269466.999)


def test_initialize_corsika_telescope_from_file(
    manual_corsika_dict_north,
    array_layout_north_instance,
    manual_corsika_dict_south,
    array_layout_south_instance,
    args_dict,
    io_handler,
):
    def test_one_site(instance, corsika_dict):
        array_layout_north_instance._initialize_corsika_telescope()

        for key, value in corsika_dict["corsika_sphere_radius"].items():
            assert value == instance._corsika_telescope["corsika_sphere_radius"][key]
        for key, value in corsika_dict["corsika_sphere_center"].items():
            assert value == instance._corsika_telescope["corsika_sphere_center"][key]

    test_one_site(array_layout_north_instance, manual_corsika_dict_north)
    test_one_site(array_layout_south_instance, manual_corsika_dict_south)


def test_initialize_array_layout_from_telescope_file(
    telescope_north_test_file,
    array_layout_north_instance,
    telescope_south_test_file,
    array_layout_south_instance,
    db_config,
):
    def test_one_site(instance, test_file, number_of_telescopes, label):
        instance.initialize_array_layout_from_telescope_file(test_file)
        assert number_of_telescopes == instance.get_number_of_telescopes()

        layout_2 = ArrayLayout(
            name="test_layout",
            telescope_list_file=test_file,
        )
        layout_2.convert_coordinates()
        assert number_of_telescopes == layout_2.get_number_of_telescopes()

    test_one_site(array_layout_north_instance, telescope_north_test_file, 13, "North")
    test_one_site(array_layout_south_instance, telescope_south_test_file, 51, "South")


def test_select_assets(telescope_north_with_calibration_devices_test_file):
    layout = ArrayLayout(
        name="test_layout", telescope_list_file=telescope_north_with_calibration_devices_test_file
    )

    layout.select_assets(None)
    assert len(layout._telescope_list) == 25

    layout.select_assets([])
    assert len(layout._telescope_list) == 25

    layout.select_assets(["MST", "SST"])
    assert len(layout._telescope_list) == 9

    layout.select_assets(["NOT_AN_ASSET", "ALSO_NOT_AN_ASSET"])
    assert len(layout._telescope_list) == 0


def test_add_tel(
    telescope_north_test_file,
    array_layout_north_instance,
    telescope_south_test_file,
    array_layout_south_instance,
):
    def test_one_site(instance, test_file, altitude):
        instance.initialize_array_layout_from_telescope_file(telescope_list_file=test_file)
        ntel_before = instance.get_number_of_telescopes()
        instance.add_telescope("LST-00", "ground", 100.0 * u.m, 50.0 * u.m, 2177.0 * u.m)
        ntel_after = instance.get_number_of_telescopes()
        assert ntel_before + 1 == ntel_after

        instance.add_telescope("LST-00", "ground", 100.0 * u.m, 50.0 * u.m, None, 50.0 * u.m)
        assert instance._telescope_list[-1].get_altitude().value == pytest.approx(altitude)

    test_one_site(array_layout_north_instance, telescope_north_test_file, 2190.0)
    test_one_site(array_layout_south_instance, telescope_south_test_file, 2181.0)


def test_build_layout(
    array_layout_north_four_LST_instance,
    array_layout_south_four_LST_instance,
    tmp_test_directory,
    db_config,
    io_handler,
    capfd,
):
    def test_one_site(
        layout, site_label, add_geocode=False, asset_code=False, sequence_number=False
    ):
        layout.add_telescope(
            telescope_name="LST-01",
            crs_name="ground",
            xx=57.5 * u.m,
            yy=57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )
        layout.add_telescope(
            telescope_name="LST-02",
            crs_name="ground",
            xx=-57.5 * u.m,
            yy=57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )
        layout.add_telescope(
            telescope_name="LST-02",
            crs_name="ground",
            xx=57.5 * u.m,
            yy=-57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )
        layout.add_telescope(
            telescope_name="LST-04",
            crs_name="ground",
            xx=-57.5 * u.m,
            yy=-57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )
        if add_geocode:
            for tel in layout._telescope_list:
                tel.geo_code = "test_geo_code"
        if asset_code:
            for tel in layout._telescope_list:
                tel.asset_code = "test_asset_code"
        if sequence_number:
            for tel in layout._telescope_list:
                tel.sequence_number = 1

        layout.convert_coordinates()
        # test that certain keywords appear in printout
        layout.print_telescope_list()
        captured_printout, _ = capfd.readouterr()
        substrings_to_check = ["LST", "Ground", "UTM", "Longitude"]
        assert all(substring in captured_printout for substring in substrings_to_check)

        layout.print_telescope_list("ground", corsika_z=True)
        captured_printout, _ = capfd.readouterr()
        assert "position_z" in captured_printout

        _table = layout.export_telescope_list_table(crs_name="ground")
        _export_file = tmp_test_directory / "test_layout.ecsv"
        _table.write(_export_file, format="ascii.ecsv", overwrite=True)

        assert isinstance(_table, QTable)

        if add_geocode:
            assert "geo_code" in _table.colnames
            return
        if asset_code and not sequence_number:
            assert "asset_code" not in _table.colnames
            return
        if not asset_code and sequence_number:
            assert "sequence_number" not in _table.colnames
            return
        if asset_code and sequence_number:
            assert "asset_code" in _table.colnames
            assert "sequence_number" in _table.colnames
            return

        # Building a second layout from the file exported by the first one
        layout_2 = ArrayLayout(site=site_label, mongo_db_config=db_config, name="test_layout_2")
        layout_2.initialize_array_layout_from_telescope_file(_export_file)

        assert 4 == layout_2.get_number_of_telescopes()
        assert layout_2._array_center.get_altitude().value == pytest.approx(
            layout._array_center.get_altitude().value, 1.0e-2
        )

    test_one_site(array_layout_north_four_LST_instance, "North")
    test_one_site(array_layout_south_four_LST_instance, "South")
    test_one_site(array_layout_north_four_LST_instance, "North", add_geocode=True)
    test_one_site(
        array_layout_north_four_LST_instance, "North", asset_code=True, sequence_number=False
    )
    test_one_site(
        array_layout_north_four_LST_instance, "North", asset_code=False, sequence_number=True
    )
    test_one_site(
        array_layout_north_four_LST_instance, "North", asset_code=True, sequence_number=True
    )


def test_converting_center_coordinates_north(array_layout_north_four_LST_instance):
    layout = array_layout_north_four_LST_instance

    _lat, _lon, _ = layout._array_center.get_coordinates("mercator")
    assert _lat.value == pytest.approx(28.7621661)
    assert _lon.value == pytest.approx(-17.8920302)

    _east, _north, _ = layout._array_center.get_coordinates("utm")
    assert _north.value == pytest.approx(3185066.278)
    assert _east.value == pytest.approx(217611.227)

    assert layout._array_center.get_altitude().value == pytest.approx(2177.0)


def test_converting_center_coordinates_south(array_layout_south_four_LST_instance):
    layout = array_layout_south_four_LST_instance

    _lat, _lon, _ = layout._array_center.get_coordinates("mercator")
    assert _lat.value == pytest.approx(-24.68342915473787)
    assert _lon.value == pytest.approx(-70.31634499364885)

    _east, _north, _ = layout._array_center.get_coordinates("utm")
    assert _north.value == pytest.approx(7269466.0)
    assert _east.value == pytest.approx(366822.0)

    assert layout._array_center.get_altitude().value == pytest.approx(2162.35)


def test_get_corsika_input_list(
    array_layout_north_four_LST_instance, array_layout_south_four_LST_instance, caplog
):
    def test_one_site(layout):
        layout.add_telescope(
            telescope_name="LST-01",
            crs_name="ground",
            xx=57.5 * u.m,
            yy=57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )
        corsika_input_list = layout.get_corsika_input_list()

        assert (
            corsika_input_list
            == "TELESCOPE\t 57.500E2\t 57.500E2\t 0.000E2\t 12.500E2\t # LST-01\n"
        )

        del layout._corsika_telescope["corsika_sphere_radius"]
        with caplog.at_level(logging.ERROR):
            with pytest.raises(KeyError):
                layout.get_corsika_input_list()
        assert "Missing definition of CORSIKA sphere radius" in caplog.text

        layout._corsika_telescope["corsika_sphere_radius"] = {"LST": 16.0 * u.m}
        del layout._corsika_telescope["corsika_obs_level"]
        with caplog.at_level(logging.ERROR):
            with pytest.raises(KeyError):
                layout.get_corsika_input_list()
        assert "Missing definition of CORSIKA sphere center / obs_level" in caplog.text

    test_one_site(array_layout_north_four_LST_instance)
    test_one_site(array_layout_south_four_LST_instance)


def test_altitude_from_corsika_z(
    array_layout_north_four_LST_instance, array_layout_south_four_LST_instance
):
    def test_one_site(instance, result1, result2):
        instance.add_telescope(
            telescope_name="LST-01",
            crs_name="ground",
            xx=57.5 * u.m,
            yy=57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )

        assert instance._altitude_from_corsika_z(
            pos_z=5.0 * u.m, altitude=None, tel_name="LST-01"
        ).value == pytest.approx(result1)
        assert instance._altitude_from_corsika_z(
            pos_z=None, altitude=2348.0 * u.m, tel_name="LST-01"
        ).value == pytest.approx(result2)
        with pytest.raises(TypeError):
            instance._altitude_from_corsika_z(5.0, None, "LST-01")
        assert np.isnan(instance._altitude_from_corsika_z(None, None, "LST-01"))

    test_one_site(array_layout_north_four_LST_instance, 2147.0, 206.0)
    test_one_site(array_layout_south_four_LST_instance, 2136.0, 217.0)


def test_include_radius_into_telescope_table(telescope_north_test_file, telescope_south_test_file):
    values_from_file_north = [20.29, -352.48, 62.00, 9.15]
    values_from_file_south = [-149.32, 76.45, 28.00]

    def test_one_site(test_file, values_from_file, mst_10_name):
        telescope_table = data_reader.read_table_from_file(test_file, validate=False)
        telescope_table_with_radius = ArrayLayout.include_radius_into_telescope_table(
            telescope_table
        )
        keys = ["position_x", "position_y", "position_z", "radius"]
        mst_10_index = telescope_table_with_radius["telescope_name"] == mst_10_name
        for key, value_manual in zip(keys, values_from_file):
            assert (
                pytest.approx(telescope_table_with_radius[mst_10_index][key].value[0], 1e-2)
                == value_manual
            )

    test_one_site(telescope_north_test_file, values_from_file_north, "MST-10")
    test_one_site(telescope_south_test_file, values_from_file_south, "MST-10")


def test_from_corsika_file_to_dict(
    array_layout_north_instance, manual_corsika_dict_north, db, io_handler
):
    def run(corsika_dict):
        for key, value in corsika_dict.items():
            if isinstance(value, dict):
                for tel_type, subvalue in value.items():
                    assert subvalue == manual_corsika_dict_north[key][tel_type]
            else:
                assert value == manual_corsika_dict_north[key]

    corsika_dict = array_layout_north_instance._from_corsika_file_to_dict()
    run(corsika_dict)

    test_file_name = "corsika_parameters_2.yml"
    db.export_file_db(
        db_name="test-data",
        dest=io_handler.get_output_directory(sub_dir="parameters", dir_type="test"),
        file_name=test_file_name,
    )

    corsika_config_file = gen.find_file(
        test_file_name, io_handler.get_output_directory(sub_dir="parameters", dir_type="test")
    )
    corsika_dict = array_layout_north_instance._from_corsika_file_to_dict(
        file_name=corsika_config_file
    )
    run(corsika_dict)

    with pytest.raises(FileNotFoundError):
        corsika_dict = array_layout_north_instance._from_corsika_file_to_dict(
            file_name="file_doesnt_exist.yml"
        )

    _save_db_config = array_layout_north_instance.mongo_db_config
    array_layout_north_instance.mongo_db_config = None
    with pytest.raises(ValueError):
        corsika_dict = array_layout_north_instance._from_corsika_file_to_dict(
            file_name=corsika_config_file
        )
    array_layout_north_instance.mongo_db_config = _save_db_config
    array_layout_north_instance.site = None
    with pytest.raises(ValueError):
        corsika_dict = array_layout_north_instance._from_corsika_file_to_dict(
            file_name=corsika_config_file
        )


def test_initialize_sphere_parameters(array_layout_north_instance):
    w_1 = {"LST": 16.0 * u.m}
    t_1 = array_layout_north_instance._initialize_sphere_parameters(w_1)
    assert pytest.approx(t_1["LST"].value, 0.1) == 16.0
    assert t_1["LST"].unit == u.m

    w_2 = {"LST": "16. m", "MST": 12.0 * u.m}
    t_2 = array_layout_north_instance._initialize_sphere_parameters(w_2)
    assert pytest.approx(t_2["LST"].value, 0.1) == 16.0
    assert t_2["LST"].unit == u.m
    assert pytest.approx(t_2["MST"].value, 0.1) == 12.0
    assert t_2["MST"].unit == u.m

    w_3 = {"LST": {"value": 10.0, "unit": "m"}}
    t_3 = array_layout_north_instance._initialize_sphere_parameters(w_3)
    assert pytest.approx(t_3["LST"].value, 0.1) == 10.0
    assert t_3["LST"].unit == u.m

    w_4 = {"LST": 16.0}
    with pytest.raises(TypeError):
        array_layout_north_instance._initialize_sphere_parameters(w_4)


def test_initialize_corsika_telescope_from_dict(
    array_layout_north_instance,
    manual_corsika_dict_north,
    array_layout_south_instance,
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

    test_one_site(array_layout_north_instance, manual_corsika_dict_north)
    test_one_site(array_layout_south_instance, manual_corsika_dict_south)

    manual_corsika_dict_north["corsika_obs_level"] = "not_a_quantity"
    array_layout_north_instance._initialize_corsika_telescope_from_dict(manual_corsika_dict_north)
    assert np.isnan(array_layout_north_instance._corsika_telescope["corsika_obs_level"])

    # no errors should be raised for missing fields
    manual_corsika_dict_south.pop("corsika_obs_level")
    manual_corsika_dict_south.pop("corsika_sphere_radius")
    array_layout_south_instance._initialize_corsika_telescope_from_dict(manual_corsika_dict_south)


def test_assign_unit_to_quantity(array_layout_north_instance):
    quantity = array_layout_north_instance._assign_unit_to_quantity(10, u.m)
    assert quantity == 10 * u.m

    quantity = array_layout_north_instance._assign_unit_to_quantity(1000 * u.cm, u.m)
    assert quantity == 10 * u.m

    with pytest.raises(u.UnitConversionError):
        array_layout_north_instance._assign_unit_to_quantity(1000 * u.TeV, u.m)


def test_try_set_altitude(
    array_layout_north_instance,
    telescope_north_test_file,
    array_layout_south_instance,
    telescope_south_test_file,
):
    obs_level_north = 2158.0
    manual_z_positions_north = [43.00, 32.00, 28.70, 32.00, 50.3, 24.0]
    corsika_sphere_center_north = [16.0, 16.0, 16.0, 16.0, 9.0, 9.0]

    obs_level_south = 2147.0
    manual_z_positions_south = [34.30, 29.40, 31.00, 33.10, 24.35, 31.00]
    corsika_sphere_center_south = [16.0, 16.0, 16.0, 16.0, 9.0, 9.0]

    def test_one_site(test_file, instance, obs_level, manual_z_positions, corsika_sphere_center):
        table = data_reader.read_table_from_file(test_file, validate=False)
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
        array_layout_north_instance,
        obs_level_north,
        manual_z_positions_north,
        corsika_sphere_center_north,
    )
    test_one_site(
        telescope_south_test_file,
        array_layout_south_instance,
        obs_level_south,
        manual_z_positions_south,
        corsika_sphere_center_south,
    )


def test_try_set_coordinate(
    array_layout_north_instance,
    telescope_north_test_file,
    array_layout_south_instance,
    telescope_south_test_file,
):
    manual_xx_north = [-70.99, -35.38, 75.22, 30.78, -211.61, -153.34]
    manual_yy_north = [-52.08, 66.14, 50.45, -64.51, 5.67, 169.04]

    manual_xx_south = [0.00, 1.45, -1.45, 150.79, 149.35, 152.38]
    manual_yy_south = [0.00, 151.07, -151.07, 73.60, -76.48, 238.67]

    def test_one_site(instance, test_file, manual_xx, manual_yy):
        table = data_reader.read_table_from_file(test_file, validate=False)
        for step, row in enumerate(table[:6]):
            tel = instance._load_telescope_names(row)
            instance._try_set_coordinate(row, tel, table, "ground", "position_x", "position_y")
            assert pytest.approx(tel.crs["ground"]["xx"]["value"], 0.1) == manual_xx[step]
            assert pytest.approx(tel.crs["ground"]["yy"]["value"], 0.1) == manual_yy[step]

        del table["telescope_name"]
        for step, row in enumerate(table[:6]):
            with pytest.raises(
                InvalidTelescopeListFile,
                match="Missing required row with telescope_name or asset_code/sequence_number",
            ):
                tel = instance._load_telescope_names(row)

    test_one_site(
        array_layout_north_instance, telescope_north_test_file, manual_xx_north, manual_yy_north
    )
    test_one_site(
        array_layout_south_instance, telescope_south_test_file, manual_xx_south, manual_yy_south
    )


def test_get_corsika_sphere_center(telescope_north_test_file):
    layout = ArrayLayout(telescope_list_file=telescope_north_test_file)

    assert layout._get_corsika_sphere_center("LST") == 16.0 * u.m

    assert layout._get_corsika_sphere_center("not_a_telescope") == 0.0 * u.m
    assert layout._get_corsika_sphere_center("") == 0.0 * u.m


def test_len(telescope_north_test_file):
    layout = ArrayLayout(telescope_list_file=telescope_north_test_file)
    assert len(layout) == 13


def test_getitem(telescope_north_test_file):
    layout = ArrayLayout(telescope_list_file=telescope_north_test_file)

    assert layout[0].name == "LST-01"


def test_load_telescope_list(
    telescope_north_test_file,
    telescope_north_utm_test_file,
    telescope_north_mercator_test_file,
    db_config,
    io_handler,
):
    _ground_table = QTable.read(telescope_north_test_file, format="ascii.ecsv")
    _ground_layout = ArrayLayout(mongo_db_config=db_config, site="North")
    _ground_layout._load_telescope_list(_ground_table)
    assert len(_ground_layout._telescope_list) == 13
    assert _ground_layout._telescope_list[0].crs["ground"]["xx"]["value"] == pytest.approx(
        _ground_table["position_x"][0].value
    )
    assert _ground_layout._telescope_list[0].crs["ground"]["yy"]["value"] == pytest.approx(
        _ground_table["position_y"][0].value
    )

    _utm_table = QTable.read(telescope_north_utm_test_file, format="ascii.ecsv")
    _utm_layout = ArrayLayout(mongo_db_config=db_config, site="North")
    _utm_layout._load_telescope_list(_utm_table)
    assert len(_utm_layout._telescope_list) == 13
    assert _utm_layout._telescope_list[0].crs["utm"]["xx"]["value"] == pytest.approx(
        _utm_table["utm_east"][0].value
    )
    assert _utm_layout._telescope_list[0].crs["utm"]["yy"]["value"] == pytest.approx(
        _utm_table["utm_north"][0].value
    )

    _mercator_table = QTable.read(telescope_north_mercator_test_file, format="ascii.ecsv")
    _mercator_layout = ArrayLayout(mongo_db_config=db_config, site="North")
    _mercator_layout._load_telescope_list(_mercator_table)
    assert len(_mercator_layout._telescope_list) == 13
    assert _mercator_layout._telescope_list[0].crs["mercator"]["xx"]["value"] == pytest.approx(
        _mercator_table["latitude"][0].value
    )
    assert _mercator_layout._telescope_list[0].crs["mercator"]["yy"]["value"] == pytest.approx(
        _mercator_table["longitude"][0].value
    )


def test_export_telescope_list_table(telescope_north_test_file, telescope_north_utm_test_file):
    layout = ArrayLayout(telescope_list_file=telescope_north_test_file)
    table = layout.export_telescope_list_table(crs_name="ground", corsika_z=False)
    assert isinstance(table, QTable)

    assert "telescope_name" in table.colnames
    assert "geo_code" in table.colnames
    assert "altitude" in table.colnames
    assert "sequence_number" not in table.colnames

    table_cors = layout.export_telescope_list_table(crs_name="ground", corsika_z=True)
    assert "position_z" in table_cors.colnames

    layout_utm = ArrayLayout(telescope_list_file=telescope_north_utm_test_file)
    table_utm = layout_utm.export_telescope_list_table(crs_name="utm", corsika_z=False)
    assert "asset_code" in table_utm.colnames
    assert "sequence_number" in table_utm.colnames
    assert "geo_code" in table_utm.colnames
    assert "telescope_name" not in table_utm.colnames

    layout_utm._telescope_list = []
    try:
        table_utm = layout_utm.export_telescope_list_table(crs_name="utm", corsika_z=False)
    except IndexError:
        pytest.fail("IndexError raised")
