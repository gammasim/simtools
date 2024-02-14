#!/usr/bin/python3

import logging

import pytest

from simtools.utils import names

logging.getLogger().setLevel(logging.DEBUG)


def test_get_list_of_telescope_types():
    assert names.get_list_of_telescope_types(array_element_class="telescope", site=None) == [
        "LSTN",
        "MSTN",
        "LSTS",
        "MSTS",
        "SSTS",
        "SCTS",
    ]

    assert names.get_list_of_telescope_types(array_element_class="telescope", site="North") == [
        "LSTN",
        "MSTN",
    ]

    assert names.get_list_of_telescope_types(array_element_class="telescope", site="South") == [
        "LSTS",
        "MSTS",
        "SSTS",
        "SCTS",
    ]

    assert "ILLN" in names.get_list_of_telescope_types(
        array_element_class="calibration", site="North"
    )


def test_validate_name():
    with_lists = {
        "South": ["paranal", "south", "cta-south", "ctao-south", "s"],
        "North": ["lapalma", "north", "cta-north", "ctao-north", "n"],
    }

    for key, value in with_lists.items():
        for _site in value:
            assert key == names._validate_name(_site, with_lists)

    with pytest.raises(ValueError, match=r"Invalid name Aar"):
        names._validate_name("Aar", with_lists)

    with_lists_in_dicts = {
        "LSTN": ["LSTN", "lstn"],
        "MSTS": ["MSTN", "mstn"],
    }
    for key, value in with_lists_in_dicts.items():
        for _tel in value:
            assert key == names._validate_name(_tel, with_lists_in_dicts)


def test_validate_telescope_id_name(caplog):
    _test_ids = {
        "1": "01",
        "01": "01",
        "5": "05",
        "55": "55",
        "Design": "design",
        "DESIGN": "design",
        "TEST": "test",
    }
    for key, value in _test_ids.items():
        assert value == names.validate_telescope_id_name(key)

    assert "01" == names.validate_telescope_id_name(1)
    assert "11" == names.validate_telescope_id_name(11)

    for _id in ["no_id", "D2345"]:
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                names.validate_telescope_id_name(_id)
            assert f"Invalid telescope ID name {_id}" in caplog.text


def test_validate_site_name():
    for key, value in names.site_names.items():
        for test_name in value:
            assert key == names.validate_site_name(test_name)
    with pytest.raises(ValueError):
        names.validate_site_name("Not a site")


def test_validate_array_layout_name():
    for key, value in names.array_layout_names.items():
        for test_name in value:
            assert key == names.validate_array_layout_name(test_name)
    with pytest.raises(ValueError):
        names.validate_array_layout_name("Not a layout")


def test_validate_telescope_name():
    telescopes = {
        "LSTN-Design": "LSTN-design",
        "LSTN-TEST": "LSTN-test",
        "LSTN-01": "LSTN-01",
        "SSTS-01": "SSTS-01",
    }

    for key, value in telescopes.items():
        logging.getLogger().info(f"Validating {key}")
        new_name = names.validate_telescope_name(key)
        logging.getLogger().info(f"New name {new_name}")
        assert value == new_name

    with pytest.raises(ValueError):
        names.validate_telescope_name("South-MST-FlashCam-D")
    with pytest.raises(ValueError):
        names.validate_telescope_name("LSTN")


def test_get_telescope_name_from_type_site_id():
    assert "LSTN-01" == names.get_telescope_name_from_type_site_id("LST", "North", "01")
    assert "LSTN-01" == names.get_telescope_name_from_type_site_id("LST", "North", "1")
    assert "LSTS-01" == names.get_telescope_name_from_type_site_id("LST", "South", "01")


def test_get_site_from_telescope_name():
    assert "North" == names.get_site_from_telescope_name("MSTN")
    assert "North" == names.get_site_from_telescope_name("MSTN-05")
    assert "South" == names.get_site_from_telescope_name("MSTS-05")
    with pytest.raises(ValueError):
        names.get_site_from_telescope_name("LSTW")


def test_get_class_from_telescope_name():
    assert "telescope" == names.get_class_from_telescope_name("LSTN-01")
    assert "calibration" == names.get_class_from_telescope_name("ILLS-01")
    with pytest.raises(ValueError):
        names.get_site_from_telescope_name("SATW")


def test_validate_model_version_name():
    model_version = names.validate_model_version_name("p4")

    assert model_version == "prod4"

    with pytest.raises(ValueError):
        names.validate_model_version_name("p0")


def test_sanitize_name():
    assert names.sanitize_name("y_edges unit") == "y_edges_unit"
    assert names.sanitize_name("Y_EDGES UNIT") == "y_edges_unit"
    assert names.sanitize_name("123name") == "_123name"
    assert names.sanitize_name("na!@#$%^&*()me") == "na__________me"
    assert names.sanitize_name("!validName") == "_validname"

    with pytest.raises(ValueError):
        names.sanitize_name("")


def test_get_telescope_type_from_telescope_name():
    assert names.get_telescope_type_from_telescope_name("LSTN-01") == "LSTN"
    assert names.get_telescope_type_from_telescope_name("MSTN-02") == "MSTN"
    assert names.get_telescope_type_from_telescope_name("SSTS-27") == "SSTS"
    assert names.get_telescope_type_from_telescope_name("SCTS-27") == "SCTS"
    assert names.get_telescope_type_from_telescope_name("MAGIC-2") == "MAGIC"
    assert names.get_telescope_type_from_telescope_name("VERITAS-4") == "VERITAS"
    for _name in ["", "01", "Not_a_telescope", "LST", "MST"]:
        with pytest.raises(ValueError):
            names.get_telescope_type_from_telescope_name(_name)


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


def test_simtel_telescope_config_file_name():
    assert (
        names.simtel_telescope_config_file_name(
            "South", "LST-1", "prod5", label=None, extra_label=None
        )
        == "CTA-South-LST-1-prod5.cfg"
    )
    assert (
        names.simtel_telescope_config_file_name(
            "South", "LST-1", "prod5", label="A", extra_label=None
        )
        == "CTA-South-LST-1-prod5_A.cfg"
    )
    assert (
        names.simtel_telescope_config_file_name(
            "South", "LST-1", "prod5", label="A", extra_label="B"
        )
        == "CTA-South-LST-1-prod5_A_B.cfg"
    )


def test_simtel_array_config_file_name():
    assert (
        names.simtel_array_config_file_name(
            array_name="4LSTs", site="South", model_version="prod5", label=None
        )
        == "CTA-4LSTs-South-prod5.cfg"
    )
    assert (
        names.simtel_array_config_file_name(
            array_name="4LSTs", site="South", model_version="prod5", label="A"
        )
        == "CTA-4LSTs-South-prod5_A.cfg"
    )


def test_simtel_single_mirror_list_file_name():
    assert (
        names.simtel_single_mirror_list_file_name(
            site="South",
            telescope_model_name="LST-1",
            model_version="prod5",
            mirror_number=5,
            label=None,
        )
        == "CTA-single-mirror-list-South-LST-1-prod5-mirror5.dat"
    )
    assert (
        names.simtel_single_mirror_list_file_name(
            site="South",
            telescope_model_name="LST-1",
            model_version="prod5",
            mirror_number=5,
            label="A",
        )
        == "CTA-single-mirror-list-South-LST-1-prod5-mirror5_A.dat"
    )


def test_layout_telescope_list_file_name():
    assert (
        names.layout_telescope_list_file_name(
            name="Alpha",
            label=None,
        )
        == "telescope_positions-Alpha.ecsv"
    )
    assert (
        names.layout_telescope_list_file_name(
            name="Alpha",
            label="sub_MST",
        )
        == "telescope_positions-Alpha_sub_MST.ecsv"
    )


def test_ray_tracing_file_name():
    assert (
        names.ray_tracing_file_name(
            site="South",
            telescope_model_name="LST-1",
            source_distance=10.5,
            zenith_angle=45.0,
            off_axis_angle=2.5,
            mirror_number=3,
            label="instance1",
            base="Photons",
        )
        == "Photons-South-LST-1-d10.5-za45.0-off2.500_mirror3_instance1.lis"
    )

    assert (
        names.ray_tracing_file_name(
            site="South",
            telescope_model_name="LST-1",
            source_distance=10.5,
            zenith_angle=45.0,
            off_axis_angle=2.5,
            mirror_number=None,
            label=None,
            base="log",
        )
        == "log-South-LST-1-d10.5-za45.0-off2.500.log"
    )


def test_ray_tracing_results_file_name():
    assert (
        names.ray_tracing_results_file_name(
            site="South",
            telescope_model_name="LST-1",
            source_distance=10.5,
            zenith_angle=45.0,
            label="instance1",
        )
        == "ray-tracing-South-LST-1-d10.5-za45.0_instance1.ecsv"
    )
    assert (
        names.ray_tracing_results_file_name(
            site="South",
            telescope_model_name="LST-1",
            source_distance=10.5,
            zenith_angle=45.0,
            label=None,
        )
        == "ray-tracing-South-LST-1-d10.5-za45.0.ecsv"
    )


def test_ray_tracing_plot_file_name():
    assert (
        names.ray_tracing_plot_file_name(
            key="d80_cm",
            site="South",
            telescope_model_name="LST-1",
            source_distance=10.5,
            zenith_angle=45.0,
            label="instance1",
        )
        == "ray-tracing-South-LST-1-d80_cm-d10.5-za45.0_instance1.pdf"
    )
    assert (
        names.ray_tracing_plot_file_name(
            key="d80_cm",
            site="South",
            telescope_model_name="LST-1",
            source_distance=10.5,
            zenith_angle=45.0,
            label=None,
        )
        == "ray-tracing-South-LST-1-d80_cm-d10.5-za45.0.pdf"
    )


def test_camera_efficiency_results_file_name():
    assert (
        names.camera_efficiency_results_file_name(
            site="South",
            telescope_model_name="LST-1",
            zenith_angle=20,
            azimuth_angle=180,
            label="test",
        )
        == "camera-efficiency-table-South-LST-1-za020deg_azm180deg_test.ecsv"
    )
    assert (
        names.camera_efficiency_results_file_name(
            site="South",
            telescope_model_name="LST-1",
            zenith_angle=20,
            azimuth_angle=180,
            label=None,
        )
        == "camera-efficiency-table-South-LST-1-za020deg_azm180deg.ecsv"
    )


def test_camera_efficiency_simtel_file_name():
    assert (
        names.camera_efficiency_simtel_file_name(
            site="South",
            telescope_model_name="LST-1",
            zenith_angle=20,
            azimuth_angle=180,
            label="test",
        )
        == "camera-efficiency-South-LST-1-za020deg_azm180deg_test.dat"
    )
    assert (
        names.camera_efficiency_simtel_file_name(
            site="South",
            telescope_model_name="LST-1",
            zenith_angle=20,
            azimuth_angle=180,
            label=None,
        )
        == "camera-efficiency-South-LST-1-za020deg_azm180deg.dat"
    )


def test_camera_efficiency_log_file_name():
    assert (
        names.camera_efficiency_log_file_name(
            site="South",
            telescope_model_name="LST-1",
            zenith_angle=20,
            azimuth_angle=180,
            label="test",
        )
        == "camera-efficiency-South-LST-1-za020deg_azm180deg_test.log"
    )
    assert (
        names.camera_efficiency_log_file_name(
            site="South",
            telescope_model_name="LST-1",
            zenith_angle=20,
            azimuth_angle=180,
            label=None,
        )
        == "camera-efficiency-South-LST-1-za020deg_azm180deg.log"
    )
