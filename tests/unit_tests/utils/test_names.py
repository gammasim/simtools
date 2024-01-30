#!/usr/bin/python3

import logging

import pytest

from simtools.utils import names

logging.getLogger().setLevel(logging.DEBUG)


all_telescope_names = [
    "North-LST-1",
    "North-LST-D234",
    "North-MST-FlashCam-D",
    "North-MST-NectarCam-D",
    "North-MST-Structure-D",
    "South-LST-D",
    "South-MST-FlashCam-D",
    "South-MST-Structure-D",
    "South-SCT-D",
    "South-SST-1M-D",
    "South-SST-ASTRI-D",
    "South-SST-Camera-D",
    "South-SST-GCT-D",
    "South-SST-Structure-D",
]


def test_validate_sub_system_name():
    for key, value in names.all_subsystem_names.items():
        for test_name in value:
            assert key == names.validate_sub_system_name(test_name)
    with pytest.raises(ValueError):
        names.validate_sub_system_name("Not a camera")


def test_validate_telescope_id_name(caplog):
    for _id in [1, "1", "01", "D", "D234"]:
        assert names.validate_telescope_id_name(_id) == str(_id)

    for _id in ["no_id", "D2345"]:
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                names.validate_telescope_id_name(_id)
            assert f"Invalid telescope ID name {_id}" in caplog.text


def test_validate_site_name():
    for key, value in names.all_site_names.items():
        for test_name in value:
            assert key == names.validate_site_name(test_name)
    with pytest.raises(ValueError):
        names.validate_site_name("Not a site")


def test_validate_array_layout_name():
    for key, value in names.all_array_layout_names.items():
        for test_name in value:
            assert key == names.validate_array_layout_name(test_name)
    with pytest.raises(ValueError):
        names.validate_array_layout_name("Not a layout")


def test_split_telescope_model_name():
    _class, _type, _id = names.split_telescope_model_name("MST-NectarCam-5")
    assert _class == "MST"
    assert _type == "NectarCam"
    assert _id == "5"

    _class, _type, _id = names.split_telescope_model_name("MST-NectarCam")
    assert _class == "MST"
    assert _type == "NectarCam"
    assert _id == ""

    _class, _type, _id = names.split_telescope_model_name("LST")
    assert _class == "LST"
    assert _type == "LST"
    assert _id == ""

    _class, _type, _id = names.split_telescope_model_name("LST-D234")
    assert _class == "LST"
    assert _type == "LST"
    assert _id == "D234"

    _class, _type, _id = names.split_telescope_model_name("MSTN")
    assert _class == "MST"
    assert _type == "NectarCam"
    assert _id == ""

    _class, _type, _id = names.split_telescope_model_name("MSTN-01")
    assert _class == "MST"
    assert _type == "NectarCam"
    assert _id == "01"

    _class, _type, _id = names.split_telescope_model_name("MSTS")
    assert _class == "MST"
    assert _type == "FlashCam"
    assert _id == ""

    _class, _type, _id = names.split_telescope_model_name("MSTS-01")
    assert _class == "MST"
    assert _type == "FlashCam"
    assert _id == "01"

    _class, _type, _id = names.split_telescope_model_name("LSTS-01")
    assert _class == "LST"
    assert _type == "LST"
    assert _id == "01"

    _class, _type, _id = names.split_telescope_model_name("MST-1")
    assert _class == "MST"
    assert _type == ""
    assert _id == "1"


def test_validate_telescope_name():
    telescopes = {
        "lst-1": "LST-1",
        "lst-D234": "LST-D234",
        "mst-flashcam-d": "MST-FlashCam-D",
        "mst-nectarcam-d": "MST-NectarCam-D",
        "mst-NectarCam-5": "MST-NectarCam-5",
        "MST-Structure-1": "MST-Structure-1",
        "sct-d": "SCT-D",
        "sst-1m": "SST-1M",
        "sst-astri-d": "SST-ASTRI-D",
        "SST-GCT-d": "SST-GCT-D",
        "sst-gct-d": "SST-GCT-D",
        "sst-d": "SST-D",
        "mst": "MST",
    }

    for key, value in telescopes.items():
        logging.getLogger().info(f"Validating {key}")
        new_name = names.validate_telescope_model_name(key)
        logging.getLogger().info(f"New name {new_name}")
        assert value == new_name

        _class, _, _id = names.split_telescope_model_name(value)
        assert names._is_valid_name(_class, names.all_telescope_class_names)
        if len(_id) > 0:
            assert names.validate_telescope_id_name(_id)

    with pytest.raises(ValueError):
        names.split_telescope_model_name("North-MST-FlashCam-D")


def test_get_site_from_telescope_name():
    assert "North" == names.get_site_from_telescope_name("North-LST-1")
    assert "South" == names.get_site_from_telescope_name("South-MST-FlashCam-D")
    with pytest.raises(ValueError):
        names.get_site_from_telescope_name("NorthWest-LST-1")
    assert "North" == names.get_site_from_telescope_name("MSTN")
    assert "North" == names.get_site_from_telescope_name("MSTN-05")
    assert "South" == names.get_site_from_telescope_name("MSTS-05")


def test_validate_name(caplog):
    all_site_names = {
        "South": ["paranal", "south", "cta-south", "ctao-south", "s"],
        "North": ["lapalma", "north", "cta-north", "ctao-north", "n"],
    }

    for key, value in all_site_names.items():
        for _site in value:
            assert key == names._validate_name(_site, all_site_names)

    with pytest.raises(ValueError, match=r"Invalid name Aar"):
        names._validate_name("Aar", all_site_names)


def test_is_valid_name():
    all_site_names = {
        "South": ["paranal", "south", "cta-south", "ctao-south", "s"],
        "North": ["lapalma", "north", "cta-north", "ctao-north", "n"],
    }

    for _, value in all_site_names.items():
        for _site in value:
            assert names._is_valid_name(_site, all_site_names)

    assert not names._is_valid_name("Aar", all_site_names)
    assert not names._is_valid_name("Aar", {})
    assert not names._is_valid_name(5, all_site_names)


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


def test_convert_telescope_model_name_to_yaml_name():
    assert names.convert_telescope_model_name_to_yaml_name("LST-1") == "LST"
    assert names.convert_telescope_model_name_to_yaml_name("LST-D234") == "LST"
    assert names.convert_telescope_model_name_to_yaml_name("MST-FlashCam-D") == "MST-FlashCam"
    assert names.convert_telescope_model_name_to_yaml_name("SCT-D") == "SCT"
    with pytest.raises(ValueError):
        names.convert_telescope_model_name_to_yaml_name("South-SCT-D")
    with pytest.raises(ValueError):
        names.convert_telescope_model_name_to_yaml_name("LST-NectarCam-D")


def test_validate_model_version_name():
    model_version = names.validate_model_version_name("p4")

    assert model_version == "prod4"

    with pytest.raises(ValueError):
        names.validate_model_version_name("p0")


def test_get_telescope_name_db():
    assert names.get_telescope_name_db("South", "MST", "FlashCam", "D") == "South-MST-FlashCam-D"
    assert names.get_telescope_name_db("North", "MST", "NectarCam", "7") == "North-MST-NectarCam-7"

    with pytest.raises(ValueError):
        names.get_telescope_name_db("West", "MST", "FlashCam", "D")


# TODO - this will go with the new naming convention
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
    assert names.sanitize_name("!validName") == "_validname"

    with pytest.raises(ValueError):
        names.sanitize_name("")


def test_get_telescope_class():
    assert names.get_telescope_class("LSTN-01") == "LST"
    assert names.get_telescope_class("MSTN-02") == "MST"
    assert names.get_telescope_class("SSTS-27") == "SST"
    assert names.get_telescope_class("SCTS-27") == "SCT"
    assert names.get_telescope_class("MAGIC-2") == "MAGIC"
    assert names.get_telescope_class("VERITAS-4") == "VERITAS"
    assert names.get_telescope_class("LST-") == "LST"
    assert names.get_telescope_class("MST-1") == "MST"
    for _name in ["", "01", "Not_a_telescope"]:
        with pytest.raises(ValueError):
            names.get_telescope_class(_name)


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


def test_telescope_model_name_from_array_element_id(caplog):
    tests_naming = {
        "LSTN-01": {"sub_system": "structure", "tel": "LST-1"},
        "LSTN-02": {"sub_system": "structure", "tel": "LST-D234"},
        "LSTN-03": {"sub_system": "structure", "tel": "LST-D234"},
        "LSTN-04": {"sub_system": "structure", "tel": "LST-D234"},
        "MSTN-04": {"sub_system": "camera", "tel": "MST-NectarCam-D"},
        "MSTN-03": {"sub_system": "structure", "tel": "MST-Structure-D"},
        "LSTS-01": {"sub_system": "structure", "tel": "LST-D"},
        "LSTS-04": {"sub_system": "structure", "tel": "LST-D"},
        "LSTS-03": {"sub_system": "camera", "tel": "LST-D"},
        "MSTS-25": {"sub_system": "camera", "tel": "MST-FlashCam-D"},
        "MSTS-03": {"sub_system": "structure", "tel": "MST-Structure-D"},
        "SCTS-02": {"sub_system": "structure", "tel": "SCT-D"},
        "SCTS-01": {"sub_system": "camera", "tel": "SCT-D"},
        "SSTS-32": {"sub_system": "structure", "tel": "SST-Structure-D"},
        "SSTS-34": {"sub_system": "camera", "tel": "SST-Camera-D"},
        "LSTS": {"sub_system": "camera", "tel": "LST-D"},
        "MSTS": {"sub_system": "structure", "tel": "MST-Structure-D"},
        "MSTN": {"sub_system": "camera", "tel": "MST-NectarCam-D"},
    }

    for key, value in tests_naming.items():
        assert (
            names.telescope_model_name_from_array_element_id(
                array_element_id=key,
                sub_system_name=value["sub_system"],
                available_telescopes=all_telescope_names,
            )
            == value["tel"]
        )

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(IndexError):
            names.telescope_model_name_from_array_element_id(
                array_element_id="LS",
                sub_system_name="camera",
                available_telescopes=all_telescope_names,
            )
        assert "Invalid array element ID LS" in caplog.text


def test_array_element_id_from_telescope_model_name():
    available_telescopes_north = {
        "LST-1": "LSTN-01",
        "LST-D234": "LSTN",
        "MST-NectarCam-D": "MSTN",
        "MST-NectarCam-14": "MSTN-14",
        "MST-Structure-D": "MSTN",
        "MST-Structure-3": "MSTN-03",
    }
    for key, value in available_telescopes_north.items():
        assert (
            names.array_element_id_from_telescope_model_name(site="North", telescope_model_name=key)
            == value
        )

    available_telescopes_south = {
        "LST-D": "LSTS",
        "LST-3": "LSTS-03",
        "MST-FlashCam-D": "MSTS",
        "MST-FlashCam-12": "MSTS-12",
        "MST-Structure-D": "MSTS",
        "MST-Structure-9": "MSTS-09",
        "SCT-D": "SCTS",
        "SCT-11": "SCTS-11",
        "SST-Camera-D": "SSTS",
        "SST-Camera-04": "SSTS-04",
        "SST-Camera-72": "SSTS-72",
        "SST-Structure-D": "SSTS",
        "SST-Structure-12": "SSTS-12",
    }
    for key, value in available_telescopes_south.items():
        assert (
            names.array_element_id_from_telescope_model_name(site="South", telescope_model_name=key)
            == value
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
