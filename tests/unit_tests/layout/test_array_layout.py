#!/usr/bin/python3

import json
import logging

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

from simtools.data_model import data_reader
from simtools.layout.array_layout import ArrayLayout, InvalidTelescopeListFileError

logger = logging.getLogger()


def _write_ground_position_table(
    tmp_test_directory, file_name, telescope_names, position_x, position_y
):
    table = QTable()
    table["telescope_name"] = telescope_names
    table["position_x"] = position_x * u.m
    table["position_y"] = position_y * u.m
    table["position_z"] = np.zeros(len(telescope_names)) * u.m
    table["geo_code"] = [f"geo-{index}" for index in range(len(telescope_names))]
    file_path = tmp_test_directory / file_name
    table.write(file_path, format="ascii.ecsv", overwrite=True)
    return file_path


def _write_utm_position_table(tmp_test_directory, file_name, telescope_names):
    table = QTable()
    table["asset_code"] = [name.split("-")[0] for name in telescope_names]
    table["sequence_number"] = [name.split("-")[1] for name in telescope_names]
    table["utm_east"] = np.arange(len(telescope_names)) * u.m
    table["utm_north"] = np.arange(len(telescope_names)) * u.m
    table["altitude"] = np.full(len(telescope_names), 2200.0) * u.m
    table["geo_code"] = [f"geo-{index}" for index in range(len(telescope_names))]
    file_path = tmp_test_directory / file_name
    table.write(file_path, format="ascii.ecsv", overwrite=True)
    return file_path


def _write_position_json(tmp_test_directory, file_name, parameter, instrument):
    file_path = tmp_test_directory / file_name
    file_path.write_text(
        json.dumps(
            {
                "parameter": parameter,
                "instrument": instrument,
                "site": "North",
                "value": [1.0, 2.0, 3.0],
                "unit": "m",
            }
        ),
        encoding="utf-8",
    )
    return file_path


@pytest.fixture
def array_layout_north_instance(model_version):
    return ArrayLayout(site="North", model_version=model_version, name="test_layout")


@pytest.fixture
def array_layout_south_instance(model_version):
    return ArrayLayout(site="South", model_version=model_version, name="test_layout")


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
def array_layout_north_four_lst_instance(model_version):
    return ArrayLayout(
        site="North",
        label="test_layout",
        name="LST4",
        model_version=model_version,
    )


@pytest.fixture
def array_layout_south_four_lst_instance(model_version):
    return ArrayLayout(
        site="South",
        label="test_layout",
        name="LST4",
        model_version=model_version,
    )


def test_initialize_coordinate_systems(
    north_layout_center_data_dict,
    array_layout_north_instance,
    south_layout_center_data_dict,
    array_layout_south_instance,
):
    def test_one_site(center_data_dict, instance, easting, northing):
        # set center data from database
        instance._initialize_coordinate_systems()
        _x, _y, _z = instance._array_center.get_coordinates("ground")
        assert _x.value == pytest.approx(0.0)
        assert _y.value == pytest.approx(0.0)
        assert _z.value == pytest.approx(
            instance._reference_position_dict["center_altitude"].value, 1.0e-2
        )
        _lat, _lon, _z = instance._array_center.get_coordinates("mercator")
        assert _z.value == pytest.approx(
            instance._reference_position_dict["center_altitude"].value, 1.0e-2
        )
        assert _lat.value == pytest.approx(center_data_dict["center_lat"].value, 1.0e-2)
        assert _lon.value == pytest.approx(center_data_dict["center_lon"].value, 1.0e-2)

        _e, _n, _z = instance._array_center.get_coordinates("utm")
        assert _z.value == pytest.approx(
            instance._reference_position_dict["center_altitude"].value, 1.0e-2
        )
        assert _e.value == pytest.approx(
            instance._reference_position_dict["center_easting"].value, 1.0
        )
        assert _n.value == pytest.approx(
            instance._reference_position_dict["center_northing"].value, 1.0
        )

        assert instance.name == "test_layout"

    test_one_site(
        north_layout_center_data_dict, array_layout_north_instance, 217611.227, 3185066.278
    )
    test_one_site(
        south_layout_center_data_dict, array_layout_south_instance, 366822.017, 7269466.999
    )


def test_select_assets(model_version, tmp_test_directory):
    telescope_names = (
        [f"LSTN-{index:02d}" for index in range(1, 17)]
        + [f"MSTN-{index:02d}" for index in range(1, 6)]
        + [f"SSTS-{index:02d}" for index in range(1, 5)]
    )
    telescope_list_file = _write_ground_position_table(
        tmp_test_directory,
        "north-calibration.ecsv",
        telescope_names,
        np.arange(25),
        np.arange(25),
    )
    layout = ArrayLayout(
        site="North",
        name="test_layout",
        telescope_list_file=telescope_list_file,
        model_version=model_version,
    )

    layout.select_assets(None)
    assert len(layout._telescope_list) == 25

    layout.select_assets([])
    assert len(layout._telescope_list) == 25

    layout.select_assets(["MSTN", "SSTS"])
    assert len(layout._telescope_list) == 9

    layout.select_assets(["NOT_AN_ASSET", "ALSO_NOT_AN_ASSET"])
    assert len(layout._telescope_list) == 0


def test_add_tel(array_layout_south_instance):
    def test_one_site(instance, altitude, tel_name, design_model):
        ntel_before = instance.get_number_of_telescopes()
        instance.add_telescope(
            tel_name, "ground", 100.0 * u.m, 50.0 * u.m, 2177.0 * u.m, design_model=design_model
        )
        ntel_after = instance.get_number_of_telescopes()
        assert ntel_before + 1 == ntel_after

        instance.add_telescope(
            tel_name, "ground", 100.0 * u.m, 50.0 * u.m, None, 50.0 * u.m, design_model=design_model
        )
        assert instance._telescope_list[-1].get_altitude().value == pytest.approx(altitude)

    test_one_site(array_layout_south_instance, 2181.0, "LSTS-05", "LSTS-design")


def check_table_columns(_table, add_geocode, asset_code, sequence_number):
    """Helper function for test_build_layout to check table columns."""
    if add_geocode:
        assert "geo_code" in _table.colnames
    elif asset_code and not sequence_number:
        assert "asset_code" not in _table.colnames
    elif not asset_code and sequence_number:
        assert "sequence_number" not in _table.colnames
    elif asset_code and sequence_number:
        assert "asset_code" in _table.colnames
        assert "sequence_number" in _table.colnames


def test_build_layout(
    array_layout_north_four_lst_instance,
    array_layout_south_four_lst_instance,
    tmp_test_directory,
    io_handler,
    capfd,
):
    def test_one_site(
        layout, site_label, add_geocode=False, asset_code=False, sequence_number=False
    ):
        layout.add_telescope(
            telescope_name="LST" + site_label[0] + "-01",
            crs_name="ground",
            xx=57.5 * u.m,
            yy=57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )
        layout.add_telescope(
            telescope_name="LST" + site_label[0] + "-02",
            crs_name="ground",
            xx=-57.5 * u.m,
            yy=57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )
        layout.add_telescope(
            telescope_name="LST" + site_label[0] + "-02",
            crs_name="ground",
            xx=57.5 * u.m,
            yy=-57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )
        layout.add_telescope(
            telescope_name="LST" + site_label[0] + "-04",
            crs_name="ground",
            xx=-57.5 * u.m,
            yy=-57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )
        for tel in layout._telescope_list:
            if add_geocode:
                tel.geo_code = "test_geo_code"
            if asset_code:
                tel.asset_code = "test_asset_code"
            if sequence_number:
                tel.sequence_number = 1

        layout.convert_coordinates()
        # test that certain keywords appear in printout
        layout.print_telescope_list("ground")
        captured_printout, _ = capfd.readouterr()
        substrings_to_check = ["LST", "telescope_name", "position_x"]
        assert all(substring in captured_printout for substring in substrings_to_check)

        _table = layout.export_telescope_list_table(crs_name="ground")
        _export_file = tmp_test_directory / "test_layout.ecsv"
        _table.write(_export_file, format="ascii.ecsv", overwrite=True)

        assert isinstance(_table, QTable)

        check_table_columns(_table, add_geocode, asset_code, sequence_number)

    test_one_site(array_layout_north_four_lst_instance, "North")
    test_one_site(array_layout_south_four_lst_instance, "South")
    test_one_site(array_layout_north_four_lst_instance, "North", add_geocode=True)
    test_one_site(
        array_layout_north_four_lst_instance, "North", asset_code=True, sequence_number=False
    )
    test_one_site(
        array_layout_north_four_lst_instance, "North", asset_code=False, sequence_number=True
    )
    test_one_site(
        array_layout_north_four_lst_instance, "North", asset_code=True, sequence_number=True
    )


def test_altitude_from_corsika_z(
    array_layout_north_four_lst_instance, array_layout_south_four_lst_instance
):
    def test_one_site(instance, telescope_name, result1, result2):
        instance.add_telescope(
            telescope_name=telescope_name,
            crs_name="ground",
            xx=57.5 * u.m,
            yy=57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )

        assert instance._altitude_from_corsika_z(
            pos_z=result2 * u.m, altitude=None, telescope_axis_height=16.0 * u.m
        ).value == pytest.approx(result1)
        assert instance._altitude_from_corsika_z(
            pos_z=None, altitude=result1 * u.m, telescope_axis_height=16.0 * u.m
        ).value == pytest.approx(result2)
        with pytest.raises(TypeError):
            instance._altitude_from_corsika_z(5.0, None, telescope_axis_height=16.0 * u.m)
        assert np.isnan(instance._altitude_from_corsika_z(None, None, None))

    test_one_site(array_layout_north_four_lst_instance, "LSTN-01", 2187.0, 45.0)
    test_one_site(array_layout_south_four_lst_instance, "LSTS-01", 2176.0, 45.0)


def test_try_set_coordinate(
    array_layout_north_instance,
    array_layout_south_instance,
    tmp_test_directory,
):
    manual_xx_north = [-70.99, -35.38, 75.22, 30.78, -211.61, -153.34]
    manual_yy_north = [-52.08, 66.14, 50.45, -64.51, 5.67, 169.04]

    manual_xx_south = [-20.63, 80.04, -19.39, -105.06, 0.00, 1.45]
    manual_yy_south = [-64.84, -0.77, 65.22, 1.15, 0.00, 151.07]

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
                InvalidTelescopeListFileError,
                match="Missing required row with telescope_name or asset_code/sequence_number",
            ):
                instance._load_telescope_names(row)

    north_file = _write_ground_position_table(
        tmp_test_directory,
        "north.ecsv",
        [f"LSTN-{index:02d}" for index in range(1, 7)],
        np.array(manual_xx_north),
        np.array(manual_yy_north),
    )
    south_file = _write_ground_position_table(
        tmp_test_directory,
        "south.ecsv",
        [f"LSTS-{index:02d}" for index in range(1, 7)],
        np.array(manual_xx_south),
        np.array(manual_yy_south),
    )
    test_one_site(array_layout_north_instance, north_file, manual_xx_north, manual_yy_north)
    test_one_site(array_layout_south_instance, south_file, manual_xx_south, manual_yy_south)


def test_len(model_version, tmp_test_directory):
    telescope_names = [f"LSTN-{index:02d}" for index in range(1, 14)]
    telescope_list_file = _write_ground_position_table(
        tmp_test_directory,
        "north.ecsv",
        telescope_names,
        np.arange(13),
        np.arange(13),
    )
    layout = ArrayLayout(
        telescope_list_file=telescope_list_file,
        model_version=model_version,
        site="North",
    )
    assert len(layout) == 13
    assert layout.get_number_of_telescopes() == 13


def test_getitem(model_version, tmp_test_directory):
    telescope_list_file = _write_ground_position_table(
        tmp_test_directory,
        "north.ecsv",
        ["LSTN-01"],
        np.array([0.0]),
        np.array([0.0]),
    )
    layout = ArrayLayout(
        telescope_list_file=telescope_list_file,
        model_version=model_version,
        site="North",
    )

    assert layout[0].name == "LSTN-01"


def test_export_telescope_list_table(
    model_version,
    tmp_test_directory,
):
    ground_file = _write_ground_position_table(
        tmp_test_directory,
        "north-ground.ecsv",
        [f"LSTN-{index:02d}" for index in range(1, 3)],
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
    )
    utm_file = _write_utm_position_table(
        tmp_test_directory,
        "north-utm.ecsv",
        ["LSTN-01"],
    )
    layout = ArrayLayout(
        site="North",
        model_version=model_version,
        telescope_list_file=ground_file,
    )
    table = layout.export_telescope_list_table(crs_name="ground")
    assert isinstance(table, QTable)

    assert "telescope_name" in table.colnames
    assert "geo_code" in table.colnames
    assert "position_z" in table.colnames
    assert "sequence_number" not in table.colnames

    layout_utm = ArrayLayout(
        site="North",
        model_version=model_version,
        telescope_list_file=utm_file,
    )
    table_utm = layout_utm.export_telescope_list_table(crs_name="utm")
    assert "asset_code" in table_utm.colnames
    assert "sequence_number" in table_utm.colnames
    assert "geo_code" in table_utm.colnames
    assert "telescope_name" not in table_utm.colnames

    layout_utm._telescope_list = []
    try:
        layout_utm.export_telescope_list_table(crs_name="utm")
    except IndexError:
        pytest.fail("IndexError raised")


def test_export_one_telescope_as_json(model_version, tmp_test_directory):
    ground_file = _write_position_json(
        tmp_test_directory,
        "ground.json",
        "array_element_position_ground",
        "MSTN-09",
    )
    utm_file = _write_utm_position_table(
        tmp_test_directory,
        "north-utm.ecsv",
        ["LSTN-01", "LSTN-02"],
    )
    layout = ArrayLayout(
        site="North",
        model_version=model_version,
        telescope_list_file=ground_file,
    )

    ground_dict = layout.export_one_telescope_as_json(crs_name="ground")
    assert isinstance(ground_dict, dict)
    assert ground_dict["instrument"] == "MSTN-09"
    assert ground_dict["parameter"] == "array_element_position_ground"

    utm_dict = layout.export_one_telescope_as_json(crs_name="utm")
    assert utm_dict["parameter"] == "array_element_position_utm"

    mercator_dict = layout.export_one_telescope_as_json(crs_name="mercator")
    assert mercator_dict["parameter"] == "array_element_position_mercator"

    layout_utm = ArrayLayout(
        site="North",
        model_version=model_version,
        telescope_list_file=utm_file,
    )
    with pytest.raises(ValueError, match=r"Only one telescope can be exported to json"):
        layout_utm.export_one_telescope_as_json(crs_name="ground")


def test_read_table_from_json_file(model_version, tmp_test_directory):
    ground_table_file = _write_position_json(
        tmp_test_directory,
        "ground.json",
        "array_element_position_ground",
        "MSTN-09",
    )
    layout = ArrayLayout(
        site="North",
        model_version=model_version,
        telescope_list_file=ground_table_file,
    )
    ground_table = layout._read_table_from_json_file(ground_table_file)
    assert isinstance(ground_table, QTable)
    assert "position_x" in ground_table.colnames

    utm_table_file = _write_position_json(
        tmp_test_directory,
        "utm.json",
        "array_element_position_utm",
        "MSTN-09",
    )
    utm_table = layout._read_table_from_json_file(utm_table_file)
    assert isinstance(utm_table, QTable)
    assert "utm_north" in utm_table.colnames
