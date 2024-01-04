#!/usr/bin/python3

import logging
from pathlib import Path

from simtools.model.array_model import ArrayModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_input_validation(db_config, io_handler):
    array_config_data = {
        "site": "North",
        "layout_name": "test-layout",
        "model_version": "Prod5",
        "default": {"LST": "1", "MST": "FlashCam-D"},
        "MST-05": "NectarCam-D",
    }
    am = ArrayModel(label="test", array_config_data=array_config_data, mongo_db_config=db_config)

    am.print_telescope_list()

    assert am.number_of_telescopes == 13


def test_exporting_config_files(db_config, io_handler):
    array_config_data = {
        "site": "North",
        "layout_name": "test-layout",
        "model_version": "Prod5",
        "default": {"LST": "1", "MST": "FlashCam-D"},
        "MST-05": {
            "name": "NectarCam-D",
            "camera_config_name": "NectarCam-test",
        },
    }
    am = ArrayModel(label="test", array_config_data=array_config_data, mongo_db_config=db_config)

    am.export_simtel_telescope_config_files()
    am.export_simtel_array_config_file()

    list_of_export_files = [
        "Aclylite8_tra_v2013ref.dat",
        "CTA-LST_lightguide_eff_2020-04-12_average.dat",
        "CTA-MST_lightguide_FC_weighted_average.dat",
        "CTA-North-LST-1-2020-06-28_test.cfg",
        "CTA-North-MST-FlashCam-D-2020-06-28_test.cfg",
        "CTA-North-MST-NectarCam-D-2020-06-28_test_MST-05.cfg",
        "CTA-TestLayout-North-2020-06-28_test.cfg",
        "CTA-ULTRA6-dsum-shaping-FlashCam-2a-int.dat",
        "LaPalma_coords.lis",
        "NectarCAM_lightguide_efficiency_POP_131019.dat",
        "Pulse_template_nectarCam_17042020-noshift.dat",
        "array_trigger_prod5_lapalma_extended.dat",
        "atm_trans_2158_1_3_2_0_0_0.1_0.1.dat",
        "atmprof_ecmwf_north_winter_fixed.dat",
        "camera_CTA-LST-1_analogsum21_v2020-04-14.dat",
        "camera_CTA-MST-FlashCam_patch3_digitalsum9_neweff2.dat",
        "camera_CTA-MST-NectarCam_20191120_majority-3nn.dat",
        "mirror_CTA-100_1.20-86-0.04.dat",
        "mirror_CTA-N-LST1_v2019-03-31.dat",
        "pulse_FlashCam_7dynode_v2a.dat",
        "pulse_LST_8dynode_pix6_20200204.dat",
        "qe_R12992-100-05b.dat",
        "qe_lst1_20200318_high+low.dat",
        "ref_AlSiO2HfO2.dat",
        "ref_LST_2020-04-23.dat",
        "spe_FlashCam_7dynode_v0a.dat",
        "spe_LST_2020-05-09_AP2.0e-4.dat",
        "spe_afterpulse_pdf_NectarCam_18122019.dat",
        "transmission_lst_window_No7-10_ave.dat",
    ]

    for modelfile in list_of_export_files:
        assert Path(am.get_config_directory()).joinpath(modelfile).exists()
