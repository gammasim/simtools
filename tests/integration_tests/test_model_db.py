"""Integration tests for model database functionality."""
#!/usr/bin/python3

import copy
import logging
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from simtools.db.db_handler import DatabaseHandler
from simtools.model.array_model import ArrayModel

logger = logging.getLogger()


def test_exporting_config_files(model_version):
    am = ArrayModel(
        label="test",
        site="North",
        layout_name="test_layout",
        model_version=model_version,
    )

    am.export_simtel_telescope_config_files()
    am.export_sim_telarray_config_file()

    test_cfg = "_test.cfg"
    list_of_export_files = [
        "CTA-LST_lightguide_eff_2020-04-12_average.dat",
        "CTA-North-LSTN-01" + test_cfg,
        "CTA-North-MSTN-01" + test_cfg,
        "CTA-test_layout-North" + test_cfg,
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

    for model_file in list_of_export_files:
        logger.info("Checking file: %s", model_file)
        assert Path(am.get_config_directory()).joinpath(model_file).exists()


def test_load_parameters_from_db(telescope_model_lst, mocker):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    mock_db = mocker.patch.object(DatabaseHandler, "get_model_parameters")
    telescope_copy._load_parameters_from_db()
    assert mock_db.call_count == 3

    telescope_copy.db = None
    telescope_copy._load_parameters_from_db()
    assert mock_db.call_count == 3


def test_export_model_files(telescope_model_lst, mocker):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    mock_db = mocker.patch.object(DatabaseHandler, "export_model_files")
    telescope_copy.export_model_files()
    assert telescope_copy._is_exported_model_files_up_to_date
    mock_db.assert_called_once()

    telescope_copy._added_parameter_files = ["test_file"]
    with pytest.raises(KeyError):
        telescope_copy.export_model_files()


def test_export_nsb_spectrum_to_telescope_altitude_correction_file(telescope_model_lst, mocker):
    model_directory = Path("test_model_directory")
    telescope_copy = copy.deepcopy(telescope_model_lst)

    mock_db_export = mocker.patch.object(DatabaseHandler, "export_model_files")
    mock_simulation_config_parameters = {
        "sim_telarray": {"correct_nsb_spectrum_to_telescope_altitude": {"value": "test_value"}}
    }
    telescope_copy._simulation_config_parameters = mock_simulation_config_parameters

    telescope_copy.export_nsb_spectrum_to_telescope_altitude_correction_file(model_directory)

    mock_db_export.assert_called_once_with(
        parameters={
            "nsb_spectrum_at_2200m": {
                "value": "test_value",
                "file": True,
            }
        },
        dest=model_directory,
    )


# depends on prod5; prod6 is incomplete in the DB
def test_read_two_dim_wavelength_angle(telescope_model_sst_prod5):
    tel_model = telescope_model_sst_prod5
    tel_model.write_sim_telarray_config_file()

    two_dim_file = tel_model.get_parameter_value("camera_filter")
    assert tel_model.config_file_directory.joinpath(two_dim_file).exists()
    two_dim_dist = tel_model.read_two_dim_wavelength_angle(two_dim_file)
    assert len(two_dim_dist["Wavelength"]) > 0
    assert len(two_dim_dist["Angle"]) > 0
    assert len(two_dim_dist["z"]) > 0
    assert two_dim_dist["Wavelength"][4] == pytest.approx(300)
    assert two_dim_dist["Angle"][4] == pytest.approx(28)
    assert two_dim_dist["z"][4][4] == pytest.approx(0.985199988)


def test_read_incidence_angle_distribution(telescope_model_sst):
    tel_model = telescope_model_sst
    tel_model.export_model_files()

    incidence_angle_file = tel_model.get_parameter_value("camera_filter_incidence_angle")
    assert tel_model.config_file_directory.joinpath(incidence_angle_file).exists()
    incidence_angle_dist = tel_model.read_incidence_angle_distribution(incidence_angle_file)
    assert len(incidence_angle_dist["Incidence angle"]) > 0
    assert len(incidence_angle_dist["Fraction"]) > 0
    assert incidence_angle_dist["Fraction"][
        np.nanargmin(np.abs(33.05 - incidence_angle_dist["Incidence angle"].value))
    ].value == pytest.approx(0.027980644661989726)


# depends on prod5 (no 2D camera file file in prod6)
def test_calc_average_curve(telescope_model_sst_prod5):
    tel_model = telescope_model_sst_prod5
    tel_model.write_sim_telarray_config_file()

    two_dim_file = tel_model.get_parameter_value("camera_filter")
    two_dim_dist = tel_model.read_two_dim_wavelength_angle(two_dim_file)
    incidence_angle_file = tel_model.get_parameter_value("camera_filter_incidence_angle")
    incidence_angle_dist = tel_model.read_incidence_angle_distribution(incidence_angle_file)
    average_dist = tel_model.calc_average_curve(two_dim_dist, incidence_angle_dist)
    assert average_dist["z"][
        np.nanargmin(np.abs(300 - average_dist["Wavelength"]))
    ] == pytest.approx(0.9398265298920796)


# depends on prod5 (no 2D camera file file in prod6)
def test_export_table_to_model_directory(telescope_model_sst_prod5):
    tel_model = telescope_model_sst_prod5
    tel_model.write_sim_telarray_config_file()

    two_dim_file = tel_model.get_parameter_value("camera_filter")
    two_dim_dist = tel_model.read_two_dim_wavelength_angle(two_dim_file)
    incidence_angle_file = tel_model.get_parameter_value("camera_filter_incidence_angle")
    incidence_angle_dist = tel_model.read_incidence_angle_distribution(incidence_angle_file)
    average_dist = tel_model.calc_average_curve(two_dim_dist, incidence_angle_dist)
    one_dim_file = tel_model.export_table_to_model_directory("test_average_curve.dat", average_dist)
    assert one_dim_file.exists()


def test_get_telescope_effective_focal_length(telescope_model_lst, telescope_model_sst_prod5):
    tel_model_lst = copy.deepcopy(telescope_model_lst)
    assert tel_model_lst.get_telescope_effective_focal_length("m") == pytest.approx(29.237)

    tel_model_sst = copy.deepcopy(telescope_model_sst_prod5)
    assert tel_model_sst.get_telescope_effective_focal_length("m") == pytest.approx(2.15191)

    # test zero case
    tel_model_sst.parameters["effective_focal_length"]["value"] = 0.0
    assert tel_model_sst.get_telescope_effective_focal_length("m") == pytest.approx(0.0)
    assert tel_model_sst.get_telescope_effective_focal_length("m", True) == pytest.approx(2.15)
    tel_model_sst.parameters["effective_focal_length"]["value"] = None
    assert tel_model_sst.get_telescope_effective_focal_length("m", True) == pytest.approx(2.15)


def test_export_single_mirror_list_file(telescope_model_lst, caplog, monkeypatch):
    tel_model = telescope_model_lst
    tel_model.write_sim_telarray_config_file()
    mirror_number = 1

    monkeypatch.setattr(
        tel_model,
        "_load_simtel_config_writer",
        lambda *args, **kwargs: setattr(tel_model, "simtel_config_writer", Mock()),
    )

    # Case 1: Valid mirror number
    tel_model.export_single_mirror_list_file(mirror_number, False)
    assert mirror_number in tel_model._single_mirror_list_file_paths
    tel_model.simtel_config_writer.write_single_mirror_list_file.assert_called_once()

    # Case 2: Invalid mirror number
    mirror_number = 9999
    with caplog.at_level(logging.ERROR):
        tel_model.export_single_mirror_list_file(mirror_number, False)
    assert "mirror_number > number_of_mirrors" in caplog.text
    assert mirror_number not in tel_model._single_mirror_list_file_paths
