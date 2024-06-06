#!/usr/bin/python3

import logging
from pathlib import Path

import pytest
from astropy import units as u
from astropy.table import QTable

from simtools.model.array_model import ArrayModel, InvalidArrayConfigDataError

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture()
def array_model(db_config, io_handler, model_version):
    return ArrayModel(
        label="test",
        site="North",
        layout_name="test_layout",
        mongo_db_config=db_config,
        model_version=model_version,
    )


@pytest.fixture()
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


def test_input_validation(array_model):
    am = array_model
    am.print_telescope_list()
    assert am.number_of_telescopes == 13


def test_site(array_model):
    am = array_model
    assert am.site == "North"


def test_get_single_telescope_info_from_array_config(db_config, model_version, io_handler):
    parameters_to_change = {
        "MSTN-05": {  # change MST pulse shape for testing to LST pulse shape
            "name": "MSTN-05",
            "fadc_pulse_shape": "LST_pulse_shape_7dynode_high_intensity_pix1s.dat",
        },
    }
    am = ArrayModel(
        label="test",
        site="North",
        layout_name="test_layout",
        parameters_to_change=parameters_to_change,
        mongo_db_config=db_config,
        model_version=model_version,
    )

    assert am._get_single_telescope_info_from_array_config("LSTN-01", parameters_to_change) == {}
    assert am._get_single_telescope_info_from_array_config("MSTN-05", parameters_to_change) == {
        "fadc_pulse_shape": "LST_pulse_shape_7dynode_high_intensity_pix1s.dat"
    }

    parameters_missing_name = {
        "MSTN-05": {  # change MST pulse shape for testing to LST pulse shape
            "fadc_pulse_shape": "LST_pulse_shape_7dynode_high_intensity_pix1s.dat",
        },
    }
    with pytest.raises(
        InvalidArrayConfigDataError, match="ArrayConfig has no name for a telescope"
    ):
        ArrayModel(
            label="test",
            site="North",
            layout_name="test_layout",
            parameters_to_change=parameters_missing_name,
            mongo_db_config=db_config,
            model_version=model_version,
        )

    parameters_with_string = {
        "MSTN-05": "a string",
    }
    am_with_string = ArrayModel(
        label="test",
        site="North",
        layout_name="test_layout",
        parameters_to_change=parameters_with_string,
        mongo_db_config=db_config,
        model_version=model_version,
    )
    assert (
        am_with_string._get_single_telescope_info_from_array_config(
            "MSTN-05", parameters_with_string
        )
        == {}
    )

    invalid_parameters = {
        "MSTN-05": 5.0,
    }
    with pytest.raises(
        InvalidArrayConfigDataError, match="ArrayConfig has wrong input for a telescope"
    ):
        ArrayModel(
            label="test",
            site="North",
            layout_name="test_layout",
            parameters_to_change=invalid_parameters,
            mongo_db_config=db_config,
            model_version=model_version,
        )


def test_exporting_config_files(db_config, io_handler, model_version):
    am = ArrayModel(
        label="test",
        site="North",
        layout_name="test_layout",
        mongo_db_config=db_config,
        model_version=model_version,
    )

    am.export_simtel_telescope_config_files()
    am.export_simtel_array_config_file()

    list_of_export_files = [
        "CTA-LST_lightguide_eff_2020-04-12_average.dat",
        "CTA-North-LSTN-01-" + model_version + "_test.cfg",
        "CTA-North-MSTN-01-" + model_version + "_test.cfg",
        "CTA-test_layout-North-" + model_version + "_test.cfg",
        "array_coordinates_LaPalma_alpha.dat",
        "NectarCAM_lightguide_efficiency_POP_131019.dat",
        "Pulse_template_nectarCam_17042020-noshift.dat",
        "array_trigger_prod6_lapalma_4_26_0.dat",
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
        "LSTN-01", "North", 10.0 * u.m, 200.0 * u.cm, 30.0 * u.m
    ) == {
        "parameter": "array_element_position_ground",
        "instrument": "LSTN-01",
        "site": "North",
        "version": "2024-02-01",
        "value": "10.0 2.0 30.0",
        "unit": "m",
        "type": "float64",
        "applicable": True,
        "file": False,
    }


def test_set_config_file_directory(array_model, io_handler):
    am = array_model
    _config_dir_1 = am.io_handler.get_output_directory(am.label, "model")
    am._set_config_file_directory()
    assert _config_dir_1.is_dir()


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
