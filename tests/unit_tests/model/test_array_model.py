#!/usr/bin/python3

import logging
from pathlib import Path

import pytest
from astropy import units as u

from simtools.model.array_model import ArrayModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def array_model(db_config, io_handler, model_version):
    return ArrayModel(
        label="test",
        site="North",
        layout_name="test-layout",
        mongo_db_config=db_config,
        model_version=model_version,
    )


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
        layout_name="test-layout",
        parameters_to_change=parameters_to_change,
        mongo_db_config=db_config,
        model_version=model_version,
    )

    assert am._get_single_telescope_info_from_array_config("LSTN-01", parameters_to_change) == {}
    assert am._get_single_telescope_info_from_array_config("MSTN-05", parameters_to_change) == {
        "fadc_pulse_shape": "LST_pulse_shape_7dynode_high_intensity_pix1s.dat"
    }


def test_exporting_config_files(db_config, io_handler, model_version):
    am = ArrayModel(
        label="test",
        site="North",
        layout_name="test-layout",
        mongo_db_config=db_config,
        model_version=model_version,
    )

    am.export_simtel_telescope_config_files()
    am.export_simtel_array_config_file()

    list_of_export_files = [
        "CTA-LST_lightguide_eff_2020-04-12_average.dat",
        "CTA-North-LSTN-01-" + model_version + "_test.cfg",
        "CTA-North-MSTN-01-" + model_version + "_test.cfg",
        "CTA-test-layout-North-" + model_version + "_test.cfg",
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
