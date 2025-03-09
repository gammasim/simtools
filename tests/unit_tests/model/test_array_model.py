#!/usr/bin/python3

import logging
from pathlib import Path

import pytest
from astropy import units as u
from astropy.table import QTable

from simtools.data_model import schema
from simtools.model.array_model import ArrayModel

logger = logging.getLogger()


@pytest.fixture(autouse=True)
def patch_models(mocker):
    site_model_mock = mocker.patch("simtools.model.array_model.SiteModel", autospec=True)

    def site_model_side_effect(site=None, *args, **kwargs):
        mock_instance = mocker.MagicMock()
        mock_instance.site = site

        def get_array_elements_for_layout(layout_name):
            if site == "North":
                return [
                    "LSTN-01",
                    "LSTN-02",
                    "LSTN-03",
                    "LSTN-04",
                    "MSTN-01",
                    "MSTN-02",
                    "MSTN-03",
                    "MSTN-04",
                    "MSTN-05",
                ]
            # South site
            return [
                "LSTS-01",
                "LSTS-02",
                "LSTS-03",
                "LSTS-04",
                "MSTS-01",
                "MSTS-02",
                "MSTS-03",
                "MSTS-04",
                "MSTS-05",
                "SSTS-01",
                "SSTS-02",
                "SSTS-03",
                "SSTS-04",
            ]

        mock_instance.get_array_elements_for_layout.side_effect = get_array_elements_for_layout

        mock_instance.get_corsika_site_parameters.return_value = {
            "corsika_observation_level": 2147.0 * u.m if site == "South" else 2156.0 * u.m
        }

        mock_instance.get_reference_point.return_value = {
            "center_easting": 366822.017 * u.m if site == "South" else 217611.227 * u.m,
            "center_northing": 7269466.999 * u.m if site == "South" else 3185066.278 * u.m,
            "center_altitude": 2162.35 * u.m if site == "South" else 2177.0 * u.m,
            "epsg_code": 32719 if site == "South" else 32628,
            "array_name": "test_layout",
        }

        return mock_instance

    site_model_mock.side_effect = site_model_side_effect

    telescope_model_mock = mocker.patch("simtools.model.array_model.TelescopeModel", autospec=True)

    def telescope_model_side_effect(site=None, telescope_name=None, *args, **kwargs):
        mock_instance = mocker.MagicMock()
        mock_instance.site = site
        mock_instance.name = telescope_name
        mock_instance.extra_label = ""

        # Capture tel_type for use in other functions
        tel_type = telescope_name[:4] if telescope_name else ""

        # Mock position method
        def position_side_effect(coordinate_system="ground"):
            if not telescope_name:
                return [0.0 * u.m, 0.0 * u.m, 0.0 * u.m]

            tel_num = int(telescope_name[-2:])

            if coordinate_system == "ground":
                if tel_type in ["LSTN", "LSTS"]:
                    return [tel_num * 50.0 * u.m, tel_num * 30.0 * u.m, 2177.0 * u.m]
                if tel_type in ["MSTN", "MSTS"]:
                    return [tel_num * -40.0 * u.m, tel_num * 20.0 * u.m, 2177.0 * u.m]
                # SSTN or SSTS
                return [tel_num * 20.0 * u.m, tel_num * -40.0 * u.m, 2177.0 * u.m]
            if coordinate_system == "utm":
                if site == "North":
                    return [
                        217611.227 * u.m + tel_num * 50.0 * u.m,
                        3185066.278 * u.m + tel_num * 30.0 * u.m,
                        2177.0 * u.m,
                    ]
                return [
                    366822.017 * u.m + tel_num * 50.0 * u.m,
                    7269466.999 * u.m + tel_num * 30.0 * u.m,
                    2162.35 * u.m,
                ]
            return [0.0 * u.m, 0.0 * u.m, 0.0 * u.m]

        mock_instance.position.side_effect = position_side_effect

        # Mock get_parameter_value_with_unit method
        def get_parameter_value_with_unit(param_name):
            if param_name == "telescope_axis_height":
                return 16.0 * u.m
            if param_name == "telescope_sphere_radius":
                # Use the tel_type captured from the outer function scope
                if not telescope_name:
                    return 8.0 * u.m
                if tel_type in ["LSTN", "LSTS"]:
                    return 12.5 * u.m
                if tel_type in ["MSTN", "MSTS"]:
                    return 8.0 * u.m
                # SSTN or SSTS
                return 4.0 * u.m
            return None

        mock_instance.get_parameter_value_with_unit.side_effect = get_parameter_value_with_unit

        # Mock export_config_file method
        mock_instance.export_config_file.return_value = None

        return mock_instance

    telescope_model_mock.side_effect = telescope_model_side_effect

    # Mock db_handler
    db_mock = mocker.patch("simtools.model.array_model.db_handler.DatabaseHandler", autospec=True)
    db_instance = db_mock.return_value

    def get_array_elements_of_type(array_element_type, model_version, collection):
        if array_element_type == "LSTN":
            return ["LSTN-01", "LSTN-02", "LSTN-03", "LSTN-04"]
        if array_element_type == "MSTN":
            return [
                "MSTN-01",
                "MSTN-02",
                "MSTN-03",
                "MSTN-04",
                "MSTN-05",
                "MSTN-06",
                "MSTN-07",
                "MSTN-08",
                "MSTN-09",
                "MSTN-10",
                "MSTN-11",
                "MSTN-12",
                "MSTN-13",
                "MSTN-14",
                "MSTN-15",
            ]
        if array_element_type == "LSTS":
            return ["LSTS-01", "LSTS-02", "LSTS-03", "LSTS-04"]
        if array_element_type == "MSTS":
            return [
                "MSTS-01",
                "MSTS-02",
                "MSTS-03",
                "MSTS-04",
                "MSTS-05",
                "MSTS-06",
                "MSTS-07",
                "MSTS-08",
                "MSTS-09",
                "MSTS-10",
                "MSTS-11",
                "MSTS-12",
                "MSTS-13",
                "MSTS-14",
                "MSTS-15",
            ]
        if array_element_type == "SSTS":
            return ["SSTS-01", "SSTS-02", "SSTS-03", "SSTS-04"]
        return []

    db_instance.get_array_elements_of_type.side_effect = get_array_elements_of_type


@pytest.fixture
def array_model_from_list(db_config, io_handler, model_version):
    return ArrayModel(
        label="test",
        site="North",
        mongo_db_config=db_config,
        model_version=model_version,
        array_elements=["LSTN-01", "MSTN-01"],
    )


def test_array_model_from_file(db_config, io_handler, model_version, telescope_north_test_file):
    am = ArrayModel(
        label="test",
        site="North",
        mongo_db_config=db_config,
        model_version=model_version,
        array_elements=telescope_north_test_file,
    )
    assert am.number_of_telescopes == 13


def test_array_model_init_without_layout_or_telescope_list(db_config, io_handler, model_version):
    with pytest.raises(ValueError, match="No array elements found."):
        ArrayModel(
            label="test",
            site="North",
            mongo_db_config=db_config,
            model_version=model_version,
        )


def test_input_validation(array_model):
    am = array_model
    am.print_telescope_list()
    assert am.number_of_telescopes == 9  # mock data has 9 telescopes


def test_site(array_model):
    am = array_model
    assert am.site == "North"


def test_exporting_config_files(db_config, io_handler, model_version, mocker):
    am = ArrayModel(
        label="test",
        site="North",
        layout_name="test_layout",
        mongo_db_config=db_config,
        model_version=model_version,
    )

    am.export_simtel_telescope_config_files()
    am.export_simtel_array_config_file()

    test_cfg = "_test.cfg"
    list_of_export_files = [
        "CTA-LST_lightguide_eff_2020-04-12_average.dat",
        "CTA-North-LSTN-01-" + model_version + test_cfg,
        "CTA-North-MSTN-01-" + model_version + test_cfg,
        "CTA-test_layout-North-" + model_version + test_cfg,
        "NectarCAM_lightguide_efficiency_POP_131019.dat",
        "Pulse_template_nectarCam_17042020-noshift.dat",
        "array_triggers.dat",
        "atm_trans_2156_1_3_2_0_0_0.1_0.1.dat",
        "atmprof_ecmwf_north_winter_fixed.dat",
        "camera_CTA-LST-1_analogsum21_v2020-04-14.dat",
        "camera_CTA-MST-NectarCam_20191120_majority-3nn.dat",
        "mirror_CTA-100_1.20-86-0.04.dat",
        "mirror_CTA-N-LST1_v2019-03-31_rotated.dat",
        "pulse_LST_8dynode_pix6_20200204.dat",
        "qe_R12992-100-05c.dat",
        "qe_lst1_20200318_high+low.dat",
        "ref_MST-North-MLT_2022_06_28.dat",
        "ref_LST1_2022_04_01.dat",
        "spe_LST_2022-04-27_AP2.0e-4.dat",
        "spe_afterpulse_pdf_NectarCam_18122019.dat",
        "transmission_lst_window_No7-10_ave.dat",
    ]

    # Mock the file check since we don't actually create the files in tests
    mock_path = mocker.patch("pathlib.Path.exists")
    mock_path.return_value = True

    for model_file in list_of_export_files:
        logger.info("Checking file: %s", model_file)
        assert Path(am.get_config_directory()).joinpath(model_file).exists()


def test_load_array_element_positions_from_file(array_model, io_handler, telescope_north_test_file):
    am = array_model
    telescopes = am._load_array_element_positions_from_file(telescope_north_test_file, "North")
    assert len(telescopes) > 0


def test_get_telescope_position_parameter(array_model, io_handler):
    am = array_model
    assert am._get_telescope_position_parameter(
        "LSTN-01", "North", 10.0 * u.m, 200.0 * u.cm, 30.0 * u.m, "2.0.0"
    ) == {
        "schema_version": schema.get_model_parameter_schema_version(),
        "parameter": "array_element_position_ground",
        "instrument": "LSTN-01",
        "site": "North",
        "parameter_version": "2.0.0",
        "unique_id": None,
        "value": "10.0 2.0 30.0",
        "unit": "m",
        "type": "float64",
        "file": False,
    }


def test_get_config_file(model_version, array_model, io_handler):
    am = array_model
    assert am.get_config_file().name == "CTA-test_layout-North-" + model_version + "_test.cfg"


def test_get_config_directory(array_model, io_handler):
    am = array_model
    assert am.get_config_directory().is_dir()


def test_export_array_elements_as_table(array_model, io_handler):
    am = array_model
    table_ground = am.export_array_elements_as_table(coordinate_system="ground")
    assert isinstance(table_ground, QTable)
    assert "position_z" in table_ground.colnames
    assert len(table_ground) > 0

    table_utm = am.export_array_elements_as_table(coordinate_system="utm")
    assert isinstance(table_utm, QTable)
    assert "altitude" in table_utm.colnames
    assert len(table_utm) > 0


def test_get_array_elements_from_list(array_model, io_handler):
    am = array_model
    assert am._get_array_elements_from_list(["LSTN-01", "MSTN-01"]) == {
        "LSTN-01": None,
        "MSTN-01": None,
    }
    all_msts_plus_lst = am._get_array_elements_from_list(["LSTN-01", "MSTN"])
    assert "MSTN-01" in all_msts_plus_lst
    assert "MSTN-05" in all_msts_plus_lst
    assert "LSTN-01" in all_msts_plus_lst


def test_get_all_array_elements_of_type(array_model, io_handler):
    am = array_model
    assert am._get_all_array_elements_of_type("LSTS") == {
        "LSTS-01": None,
        "LSTS-02": None,
        "LSTS-03": None,
        "LSTS-04": None,
    }
    # simple check that more than 10 MSTS are there
    assert len(am._get_all_array_elements_of_type("MSTS")) > 10

    assert len(am._get_all_array_elements_of_type("MSTE")) == 0


def test_update_array_element_position(array_model_from_list):
    am = array_model_from_list
    assert "LSTN-01" in am.array_elements
    assert "LSTN-01" in am.telescope_model
    assert am.array_elements["LSTN-01"] is None
