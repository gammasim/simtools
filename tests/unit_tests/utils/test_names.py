#!/usr/bin/python3

import logging

import pytest

from simtools.utils import names

logging.getLogger().setLevel(logging.DEBUG)


def test_validate_telescope_name():
    telescopes = {"sst-d": "SST-D", "mst-flashcam-d": "MST-FlashCam-D", "sct-d": "SCT-D"}

    for key, value in telescopes.items():
        logging.getLogger().info(f"Validating {key}")
        new_name = names.validate_telescope_model_name(key)
        logging.getLogger().info(f"New name {new_name}")

        assert value == new_name


def test_validate_telescope_name_db():
    telescopes = {
        "south-sst-d": "South-SST-D",
        "north-mst-nectarcam-d": "North-MST-NectarCam-D",
        "north-lst-1": "North-LST-1",
    }

    for key, value in telescopes.items():
        logging.getLogger().info(f"Validating {key}")
        new_name = names.validate_telescope_name_db(key)
        logging.getLogger().info(f"New name {new_name}")

        assert value == new_name

    telescopes = {
        "ssss-sst-d": "SSSS-SST-D",
        "no-rth-mst-nectarcam-d": "No-rth-MST-NectarCam-D",
        "north-ls-1": "North-LS-1",
    }

    for key, value in telescopes.items():
        logging.getLogger().info(f"Validating {key}")
        with pytest.raises(ValueError):
            names.validate_telescope_name_db(key)


def test_validate_other_names():
    model_version = names.validate_model_version_name("p4")
    logging.getLogger().info(model_version)

    assert model_version == "prod4"


def test_simtools_instrument_name():
    assert names.simtools_instrument_name("South", "MST", "FlashCam", "D") == "South-MST-FlashCam-D"
    assert (
        names.simtools_instrument_name("North", "MST", "NectarCam", "7") == "North-MST-NectarCam-7"
    )

    with pytest.raises(ValueError):
        names.simtools_instrument_name("West", "MST", "FlashCam", "D")


def test_translate_corsika_to_simtools():
    corsika_pars = ["OBSLEV", "corsika_sphere_radius", "corsika_sphere_center"]
    simtools_pars = ["corsika_obs_level", "corsika_sphere_radius", "corsika_sphere_center"]
    for step, corsika_par in enumerate(corsika_pars):
        assert names.translate_corsika_to_simtools(corsika_par) == simtools_pars[step]


def test_translate_simtools_to_corsika():
    corsika_pars = ["OBSLEV", "corsika_sphere_radius", "corsika_sphere_center"]
    simtools_pars = ["corsika_obs_level", "corsika_sphere_radius", "corsika_sphere_center"]
    for step, simtools_par in enumerate(simtools_pars):
        assert names.translate_simtools_to_corsika(simtools_par) == corsika_pars[step]


def test_sanitize_name():
    assert names.sanitize_name("y_edges unit") == "y_edges_unit"
    assert names.sanitize_name("Y_EDGES UNIT") == "y_edges_unit"
    assert names.sanitize_name("123name") == "_123name"
    assert names.sanitize_name("na!@#$%^&*()me") == "na__________me"


def test_get_telescope_type():
    telescope_name = "LST-1"
    assert names.get_telescope_type(telescope_name) == "LST"
    telescope_name = "MST-2"
    assert names.get_telescope_type(telescope_name) == "MST"
    telescope_name = "SST-27"
    assert names.get_telescope_type(telescope_name) == "SST"
    telescope_name = "SCT-27"
    assert names.get_telescope_type(telescope_name) == "SCT"
    telescope_name = "MAGIC-2"
    assert names.get_telescope_type(telescope_name) == "MAGIC"
    telescope_name = "VERITAS-4"
    assert names.get_telescope_type(telescope_name) == "VERITAS"
    telescope_name = ""
    assert names.get_telescope_type(telescope_name) == ""
    telescope_name = "-1"
    assert names.get_telescope_type(telescope_name) == ""
    telescope_name = "LST-"
    assert names.get_telescope_type(telescope_name) == "LST"
    telescope_name = None
    with pytest.raises(AttributeError):
        assert names.get_telescope_type(telescope_name) == ""
    telescope_name = "Not_a_telescope"
    assert names.get_telescope_type(telescope_name) == ""


def test_camera_efficiency_names():
    """
    Test the various camera efficiency names functions
    """

    site = "South"
    telescope_model_name = "LST-1"
    zenith_angle = 20
    azimuth_angle = 180
    label = "test"
    assert (
        names.camera_efficiency_results_file_name(
            site, telescope_model_name, zenith_angle, azimuth_angle, label
        )
        == "camera-efficiency-table-South-LST-1-za020deg_azm180deg_test.ecsv"
    )
    assert (
        names.camera_efficiency_simtel_file_name(
            site, telescope_model_name, zenith_angle, azimuth_angle, label
        )
        == "camera-efficiency-South-LST-1-za020deg_azm180deg_test.dat"
    )
    assert (
        names.camera_efficiency_log_file_name(
            site, telescope_model_name, zenith_angle, azimuth_angle, label
        )
        == "camera-efficiency-South-LST-1-za020deg_azm180deg_test.log"
    )

    site = "North"
    telescope_model_name = "MST-FlashCam-D"
    zenith_angle = 40
    azimuth_angle = 0
    label = "test"
    assert (
        names.camera_efficiency_results_file_name(
            site, telescope_model_name, zenith_angle, azimuth_angle, label
        )
        == "camera-efficiency-table-North-MST-FlashCam-D-za040deg_azm000deg_test.ecsv"
    )
    assert (
        names.camera_efficiency_simtel_file_name(
            site, telescope_model_name, zenith_angle, azimuth_angle, label
        )
        == "camera-efficiency-North-MST-FlashCam-D-za040deg_azm000deg_test.dat"
    )
    assert (
        names.camera_efficiency_log_file_name(
            site, telescope_model_name, zenith_angle, azimuth_angle, label
        )
        == "camera-efficiency-North-MST-FlashCam-D-za040deg_azm000deg_test.log"
    )

    site = "South"
    telescope_model_name = "LST-1"
    zenith_angle = 20
    azimuth_angle = 180
    label = None
    assert (
        names.camera_efficiency_results_file_name(
            site, telescope_model_name, zenith_angle, azimuth_angle, label
        )
        == "camera-efficiency-table-South-LST-1-za020deg_azm180deg.ecsv"
    )
