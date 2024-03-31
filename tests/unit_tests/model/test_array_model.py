#!/usr/bin/python3

import logging
from pathlib import Path

import pytest

from simtools.model.array_model import ArrayModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def array_model(db_config, io_handler, model_version):
    array_config_data = {
        "site": "North",
        "layout_name": "test-layout",
        "default": {"LSTN": "01", "MSTN": "Design"},
        "MSTN-05": "05",
    }
    return ArrayModel(
        label="test",
        array_config_data=array_config_data,
        mongo_db_config=db_config,
        model_version=model_version,
    )


def test_input_validation(array_model):
    am = array_model
    am.print_telescope_list()
    assert am.number_of_telescopes == 13


def test_get_single_telescope_info_from_array_config(array_model):
    am = array_model

    assert am._get_single_telescope_info_from_array_config("LSTN-01") == ("LSTN-01", {})
    assert am._get_single_telescope_info_from_array_config("LSTN-02") == ("LSTN-01", {})
    assert am._get_single_telescope_info_from_array_config("MSTN-01") == ("MSTN-Design", {})
    assert am._get_single_telescope_info_from_array_config("MSTN-05") == ("MSTN-05", {})
    assert am._get_single_telescope_info_from_array_config("MSTN-15") == ("MSTN-Design", {})

    # TODO - test on parameters which change for the models


def test_exporting_config_files(db_config, io_handler, model_version):
    array_config_data = {
        "site": "North",
        "layout_name": "test-layout",
        "default": {"LSTN": "01", "MSTN": "design"},
    }
    am = ArrayModel(
        label="test",
        array_config_data=array_config_data,
        mongo_db_config=db_config,
        model_version=model_version,
    )

    am.export_simtel_telescope_config_files()
    am.export_simtel_array_config_file()

    list_of_export_files = [
        "CTA-LST_lightguide_eff_2020-04-12_average.dat",
        "CTA-North-LSTN-01-" + model_version + "_test.cfg",
        "CTA-North-MSTN-design-" + model_version + "_test.cfg",
        "CTA-TestLayout-North-" + model_version + "_test.cfg",
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
