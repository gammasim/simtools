#!/usr/bin/python3

import logging
from pathlib import Path

import pytest

from simtools.utils import names

logging.getLogger().setLevel(logging.DEBUG)


ecsv_suffix = ".ecsv"


@pytest.fixture
def invalid_name():
    return "Invalid name"


def test_model_parameters():
    assert isinstance(names.model_parameters(), dict)
    assert isinstance(names.model_parameters("Telescope"), dict)
    assert len(names.model_parameters()) > len(names.model_parameters("Telescope"))


def test_site_parameters():
    assert isinstance(names.site_parameters(), dict)
    assert "altitude" in names.site_parameters()


def test_telescope_parameters():
    assert isinstance(names.telescope_parameters(), dict)
    assert "focal_length" in names.telescope_parameters()


def test_get_list_of_array_element_types():
    assert names.get_list_of_array_element_types(array_element_class="telescopes", site=None) == [
        "LSTN",
        "LSTS",
        "MSTN",
        "MSTS",
        "MSTx",
        "SCTS",
        "SSTS",
    ]

    assert names.get_list_of_array_element_types(
        array_element_class="telescopes", site="North"
    ) == [
        "LSTN",
        "MSTN",
    ]

    assert names.get_list_of_array_element_types(
        array_element_class="telescopes", site="South"
    ) == [
        "LSTS",
        "MSTS",
        "SCTS",
        "SSTS",
    ]

    assert "ILLN" in names.get_list_of_array_element_types(
        array_element_class="calibration_devices", site="North"
    )


def test_validate_name():
    with_lists = {
        "South": ["south"],
        "North": ["north"],
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


def test_instrument_class_key_to_db_collection():
    assert "telescopes" == names.instrument_class_key_to_db_collection("Telescope")
    assert "calibration_devices" == names.instrument_class_key_to_db_collection("Calibration")
    assert "sites" == names.instrument_class_key_to_db_collection("Site")
    assert "configuration_sim_telarray" == names.instrument_class_key_to_db_collection(
        "configuration_sim_telarray"
    )
    assert "configuration_corsika" == names.instrument_class_key_to_db_collection(
        "configuration_corsika"
    )

    with pytest.raises(ValueError, match=r"^Class Not_a_class not found"):
        names.instrument_class_key_to_db_collection("Not_a_class")


def test_get_collection_name_from_parameter_name():
    assert "telescopes" == names.get_collection_name_from_parameter_name("num_gains")
    assert "sites" == names.get_collection_name_from_parameter_name("atmospheric_profile")
    assert "calibration_devices" == names.get_collection_name_from_parameter_name("laser_photons")
    assert "configuration_sim_telarray" == names.get_collection_name_from_parameter_name(
        "iobuf_maximum"
    )
    assert "configuration_corsika" == names.get_collection_name_from_parameter_name(
        "corsika_particle_kinetic_energy_cutoff"
    )
    with pytest.raises(KeyError, match=r"Parameter Not_a_parameter without schema definition"):
        names.get_collection_name_from_parameter_name("Not_a_parameter")


def test_validate_array_element_id_name(caplog):
    _test_ids = {
        "1": "01",
        "01": "01",
        "5": "05",
        "55": "55",
        "455": "455",
        "design": "design",
        "test": "test",
    }
    for key, value in _test_ids.items():
        assert value == names.validate_array_element_id_name(key)

    assert "01" == names.validate_array_element_id_name(1)
    assert "11" == names.validate_array_element_id_name(11)

    for _id in ["no_id", "D2345", "FlashCam"]:
        with pytest.raises(ValueError, match=r"^Invalid array element ID name"):
            names.validate_array_element_id_name(_id)

    assert "FlashCam" == names.validate_array_element_id_name("FlashCam", "MSTx")


def test_array_element_design_types():
    assert names.array_element_design_types(None) == ["design", "test"]
    assert names.array_element_design_types("LSTN") == ["design", "test"]
    _expected_mstx_types = ["test", "FlashCam", "NectarCam"]
    for _type in _expected_mstx_types:
        assert _type in names.array_element_design_types("MSTx")


def test_validate_site_name(invalid_name):
    for key, value in names.site_names().items():
        for test_name in value:
            assert key == names.validate_site_name(test_name)
    with pytest.raises(ValueError, match=rf"^{invalid_name}"):
        names.validate_site_name("Not a site")


def test_validate_array_element_type(invalid_name):
    assert "LSTN" == names.validate_array_element_type("LSTN")
    assert "ILLS" == names.validate_array_element_type("ILLS")

    with pytest.raises(ValueError, match=rf"^{invalid_name}"):
        names.validate_array_element_type("Not a type")
    with pytest.raises(ValueError, match=rf"^{invalid_name}"):
        names.validate_array_element_type("LSTN-01")


def test_validate_array_element_name(invalid_name):
    telescopes = {
        "LSTN-design": "LSTN-design",
        "LSTN-test": "LSTN-test",
        "LSTN-01": "LSTN-01",
        "SSTS-01": "SSTS-01",
        "OBS-North": "North",
        "MSTx-NectarCam": "MSTx-NectarCam",
    }

    for key, value in telescopes.items():
        logging.getLogger().info(f"Validating {key}")
        new_name = names.validate_array_element_name(key)
        logging.getLogger().info(f"New name {new_name}")
        assert value == new_name

    with pytest.raises(ValueError, match=rf"^{invalid_name}"):
        names.validate_array_element_name("South-MST-FlashCam-D")
    with pytest.raises(ValueError, match=rf"^{invalid_name}"):
        names.validate_array_element_name("LSTN")


def test_generate_array_element_name_from_type_site_id():
    assert "LSTN-01" == names.generate_array_element_name_from_type_site_id("LST", "North", "01")
    assert "LSTN-01" == names.generate_array_element_name_from_type_site_id("LST", "North", "1")
    assert "LSTS-01" == names.generate_array_element_name_from_type_site_id("LST", "South", "01")


def test_get_site_from_array_element_name(invalid_name):
    assert "North" == names.get_site_from_array_element_name("MSTN")
    assert "North" == names.get_site_from_array_element_name("MSTN-05")
    assert "South" == names.get_site_from_array_element_name("MSTS-05")
    with pytest.raises(ValueError, match=rf"^{invalid_name}"):
        names.get_site_from_array_element_name("LSTW")
    assert ["North", "South"] == names.get_site_from_array_element_name("MSTx")


def test_get_collection_name_from_array_element_name():
    assert "telescopes" == names.get_collection_name_from_array_element_name("LSTN-01")
    assert "telescopes" == names.get_collection_name_from_array_element_name("MSTx-FlashCam")
    assert "sites" == names.get_collection_name_from_array_element_name("North", False)
    assert "sites" == names.get_collection_name_from_array_element_name("OBS-North", False)
    assert "configuration_sim_telarray" == names.get_collection_name_from_array_element_name(
        "configuration_sim_telarray", False
    )

    with pytest.raises(ValueError, match=r"Invalid array element name configuration_sim_telarray"):
        names.get_collection_name_from_array_element_name("configuration_sim_telarray", True)

    with pytest.raises(ValueError, match=r"Invalid array element name Not_a_collection"):
        names.get_collection_name_from_array_element_name("Not_a_collection", False)


def test_sanitize_name(caplog):
    assert names.sanitize_name("y_edges unit") == "y_edges_unit"
    assert names.sanitize_name("Y_EDGES UNIT") == "y_edges_unit"
    assert names.sanitize_name("123name") == "_123name"
    assert names.sanitize_name("na!@#$%^&*()me") == "na__________me"
    assert names.sanitize_name("!validName") == "_validname"
    assert names.sanitize_name(None) is None

    with pytest.raises(ValueError, match=r"^The string  could not be sanitized."):
        names.sanitize_name("")


def test_get_array_element_type_from_name(invalid_name):
    assert names.get_array_element_type_from_name("LSTN-01") == "LSTN"
    assert names.get_array_element_type_from_name("MSTN-02") == "MSTN"
    assert names.get_array_element_type_from_name("SSTS-27") == "SSTS"
    assert names.get_array_element_type_from_name("SCTS-27") == "SCTS"
    assert names.get_array_element_type_from_name("MAGIC-2") == "MAGIC"
    assert names.get_array_element_type_from_name("VERITAS-4") == "VERITAS"
    for _name in ["", "01", "Not_a_telescope", "LST", "MST"]:
        with pytest.raises(ValueError, match=rf"^{invalid_name}"):
            names.get_array_element_type_from_name(_name)
    assert names.get_array_element_type_from_name("South") == "South"


def test_get_array_element_id_from_name(invalid_name):
    assert names.get_array_element_id_from_name("LSTN-01") == "01"
    assert names.get_array_element_id_from_name("MSTN-02") == "02"
    assert names.get_array_element_id_from_name("SSTS-27") == "27"
    assert names.get_array_element_id_from_name("SCTS-27") == "27"
    assert names.get_array_element_id_from_name("SCTS-design") == "design"
    assert names.get_array_element_id_from_name("MSTx-FlashCam") == "FlashCam"
    assert names.get_array_element_id_from_name("VERITAS-4") == "04"
    for _name in ["", "01", "design", "LST-bdesign"]:
        with pytest.raises(ValueError, match=rf"^{invalid_name}"):
            names.get_array_element_id_from_name(_name)


def test_generate_file_name_camera_efficiency():
    site = "South"
    telescope_model_name = "LSTS-01"
    zenith_angle = 20
    azimuth_angle = 180
    label = "test"

    assert (
        names.generate_file_name(
            "camera_efficiency_table",
            ecsv_suffix,
            site,
            telescope_model_name,
            zenith_angle,
            azimuth_angle,
            label=label,
        )
        == "camera_efficiency_table_South_LSTS-01_za20.0deg_azm180deg_test.ecsv"
    )

    assert (
        names.generate_file_name(
            "camera_efficiency",
            ".dat",
            site,
            telescope_model_name,
            zenith_angle,
            azimuth_angle,
            label=label,
        )
        == "camera_efficiency_South_LSTS-01_za20.0deg_azm180deg_test.dat"
    )

    assert (
        names.generate_file_name(
            "camera_efficiency",
            ".log",
            site,
            telescope_model_name,
            zenith_angle,
            azimuth_angle,
            label=label,
        )
        == "camera_efficiency_South_LSTS-01_za20.0deg_azm180deg_test.log"
    )

    site = "North"
    telescope_model_name = "MSTN"
    zenith_angle = 40
    azimuth_angle = 0
    label = "test"
    assert (
        names.generate_file_name(
            "camera_efficiency_table",
            ecsv_suffix,
            site,
            telescope_model_name,
            zenith_angle,
            azimuth_angle,
            label=label,
        )
        == "camera_efficiency_table_North_MSTN_za40.0deg_azm000deg_test.ecsv"
    )
    assert (
        names.generate_file_name(
            "camera_efficiency",
            ".dat",
            site,
            telescope_model_name,
            zenith_angle,
            azimuth_angle,
            label=label,
        )
        == "camera_efficiency_North_MSTN_za40.0deg_azm000deg_test.dat"
    )
    assert (
        names.generate_file_name(
            "camera_efficiency",
            ".log",
            site,
            telescope_model_name,
            zenith_angle,
            azimuth_angle,
            label=label,
        )
        == "camera_efficiency_North_MSTN_za40.0deg_azm000deg_test.log"
    )

    site = "South"
    telescope_model_name = "LSTS-01"
    zenith_angle = 20
    azimuth_angle = 180
    label = None
    assert (
        names.generate_file_name(
            "camera_efficiency_table",
            ecsv_suffix,
            site,
            telescope_model_name,
            zenith_angle,
            azimuth_angle,
            label=label,
        )
        == "camera_efficiency_table_South_LSTS-01_za20.0deg_azm180deg.ecsv"
    )


def test_simtel_telescope_config_file_name(model_version):
    assert (
        names.simtel_config_file_name(
            "South", model_version, telescope_model_name="LSTS-01", label=None, extra_label=None
        )
        == "CTA-South-LSTS-01-" + model_version + ".cfg"
    )
    assert (
        names.simtel_config_file_name(
            "South", model_version, telescope_model_name="LSTS-01", label="A", extra_label=None
        )
        == "CTA-South-LSTS-01-" + model_version + "_A.cfg"
    )
    assert (
        names.simtel_config_file_name(
            "South", model_version, telescope_model_name="LSTS-01", label="A", extra_label="B"
        )
        == "CTA-South-LSTS-01-" + model_version + "_A_B.cfg"
    )


def test_sim_telarray_config_file_name(model_version):
    assert (
        names.simtel_config_file_name(
            array_name="4LSTs", site="South", model_version=model_version, label=None
        )
        == "CTA-4LSTs-South-" + model_version + ".cfg"
    )
    assert (
        names.simtel_config_file_name(
            array_name="4LSTs", site="South", model_version=model_version, label="A"
        )
        == "CTA-4LSTs-South-" + model_version + "_A.cfg"
    )


def test_simtel_single_mirror_list_file_name(model_version):
    assert (
        names.simtel_single_mirror_list_file_name(
            site="South",
            telescope_model_name="LST-1",
            model_version=model_version,
            mirror_number=5,
            label=None,
        )
        == "CTA-single-mirror-list-South-LST-1-" + model_version + "-mirror5.dat"
    )
    assert (
        names.simtel_single_mirror_list_file_name(
            site="South",
            telescope_model_name="LST-1",
            model_version=model_version,
            mirror_number=5,
            label="A",
        )
        == "CTA-single-mirror-list-South-LST-1-" + model_version + "-mirror5_A.dat"
    )


def test_generate_file_name_ray_tracing():
    assert (
        names.generate_file_name(
            file_type="photons",
            suffix=".lis",
            site="South",
            telescope_model_name="LSTS-01",
            source_distance=10.5,
            zenith_angle=45.0,
            off_axis_angle=2.5,
            mirror_number=3,
            label="instance1",
        )
        == "photons_South_LSTS-01_d10.5km_za45.0deg_off2.500deg_mirror3_instance1.lis"
    )

    assert (
        names.generate_file_name(
            file_type="log",
            site="South",
            suffix=".log",
            telescope_model_name="LSTS-01",
            source_distance=10.5,
            zenith_angle=45.0,
            off_axis_angle=2.5,
            mirror_number=None,
            label=None,
        )
        == "log_South_LSTS-01_d10.5km_za45.0deg_off2.500deg.log"
    )

    assert (
        names.generate_file_name(
            file_type="ray_tracing",
            suffix=ecsv_suffix,
            site="South",
            telescope_model_name="LSTS-01",
            source_distance=10.5,
            zenith_angle=45.0,
            label="instance1",
        )
        == "ray_tracing_South_LSTS-01_d10.5km_za45.0deg_instance1.ecsv"
    )
    assert (
        names.generate_file_name(
            file_type="ray_tracing",
            suffix=ecsv_suffix,
            site="South",
            telescope_model_name="LSTS-01",
            source_distance=10.5,
            zenith_angle=45.0,
            label=None,
        )
        == "ray_tracing_South_LSTS-01_d10.5km_za45.0deg.ecsv"
    )

    assert (
        names.generate_file_name(
            file_type="ray_tracing",
            suffix=".pdf",
            extra_label="d80_cm",
            site="South",
            telescope_model_name="LSTS-01",
            source_distance=10.5,
            zenith_angle=45.0,
            label="instance1",
        )
        == "ray_tracing_South_LSTS-01_d10.5km_za45.0deg_instance1_d80_cm.pdf"
    )
    assert (
        names.generate_file_name(
            file_type="ray_tracing",
            suffix=".pdf",
            extra_label="d80_cm",
            site="South",
            telescope_model_name="LSTS-01",
            source_distance=10.5,
            zenith_angle=45.0,
            label=None,
        )
        == "ray_tracing_South_LSTS-01_d10.5km_za45.0deg_d80_cm.pdf"
    )


def test_simulation_software():
    software = names.simulation_software()
    assert isinstance(software, list)
    assert "sim_telarray" in software


def test_get_simulation_software_name_from_parameter_name():
    sim_telarray = "sim_telarray"
    assert (
        names.get_simulation_software_name_from_parameter_name(
            "focal_length", software_name=sim_telarray
        )
        == "focal_length"
    )
    assert (
        names.get_simulation_software_name_from_parameter_name(
            "telescope_axis_height", software_name=sim_telarray
        )
        is None
    )
    assert (
        names.get_simulation_software_name_from_parameter_name(
            "corsika_observation_level", software_name=sim_telarray
        )
        == "altitude"
    )
    assert (
        names.get_simulation_software_name_from_parameter_name(
            "corsika_observation_level", software_name="corsika"
        )
        == "OBSLEV"
    )
    assert (
        names.get_simulation_software_name_from_parameter_name(
            "reference_point_longitude", software_name=sim_telarray
        )
        is None  # this is not a sim_telarray parameter
    )
    assert (
        names.get_simulation_software_name_from_parameter_name(
            "reference_point_longitude", software_name="corsika"
        )
        == "reference_point_longitude"
    )
    with pytest.raises(KeyError, match=r"Parameter Not_a_parameter without schema definition"):
        names.get_simulation_software_name_from_parameter_name(
            "Not_a_parameter", software_name=sim_telarray
        )
    assert (
        names.get_simulation_software_name_from_parameter_name(
            "corsika_observation_level",
            software_name=None,
        )
        is None
    )
    assert (
        names.get_simulation_software_name_from_parameter_name(
            "reference_point_longitude",
            software_name=sim_telarray,
        )
        is None
    )
    assert (
        names.get_simulation_software_name_from_parameter_name(
            "reference_point_longitude",
            software_name=sim_telarray,
            set_meta_parameter=True,
        )
        == "longitude"
    )


def test_file_name_with_version():
    assert names.file_name_with_version(None, None) is None
    assert names.file_name_with_version("file", None) is None
    assert names.file_name_with_version(None, ".yml") is None

    assert names.file_name_with_version("file", ".yml") == Path("file.yml")
    assert names.file_name_with_version("file.json", ".yml") == Path("file.yml")

    assert names.file_name_with_version("file-5.22.0", ".yml") == Path("file-5.22.0.yml")
    assert names.file_name_with_version("file-5.0.0.json", ".yml") == Path("file-5.0.0.yml")


def test_db_collection_to_instrument_class_key():
    assert names.db_collection_to_instrument_class_key() == ["Structure", "Camera", "Telescope"]

    with pytest.raises(KeyError, match="Invalid collection name no_collection"):
        names.db_collection_to_instrument_class_key("no_collection")


def test_is_design_type():
    assert names.is_design_type("LSTN-design")
    assert names.is_design_type("MSTS-FlashCam")
    assert names.is_design_type("MSTS-NectarCam")
    assert not names.is_design_type("MSTS-22")


def test_array_element_common_identifiers():
    id_to_name, name_to_id = names.array_element_common_identifiers()
    assert isinstance(id_to_name, dict)
    assert isinstance(name_to_id, dict)
    assert len(id_to_name) > 0
    assert len(name_to_id) > 0

    # Check that the dictionaries are consistent
    for name, id_ in name_to_id.items():
        assert id_ in id_to_name
        assert id_to_name[id_] == name

    for id_, name in id_to_name.items():
        assert name in name_to_id
        assert name_to_id[name] == id_


def test_get_common_identifier_from_array_element_name():
    assert names.get_common_identifier_from_array_element_name("LSTN-01") == 1
    assert names.get_common_identifier_from_array_element_name("MSTN-08") == 12
    assert names.get_common_identifier_from_array_element_name("SSTS-03") == 121

    with pytest.raises(ValueError, match="Unknown array element name Not_a_name"):
        names.get_common_identifier_from_array_element_name("Not_a_name")

    assert names.get_common_identifier_from_array_element_name("Not_a_name", default_return=0) == 0


def test_get_array_element_name_from_common_identifier():
    _, _ = names.array_element_common_identifiers()

    # Check some known identifiers
    assert names.get_array_element_name_from_common_identifier(1) == "LSTN-01"
    assert names.get_array_element_name_from_common_identifier(12) == "MSTN-08"
    assert names.get_array_element_name_from_common_identifier(121) == "SSTS-03"

    # Check that the function raises an error for an unknown identifier
    with pytest.raises(ValueError, match="Unknown common identifier 9999"):
        names.get_array_element_name_from_common_identifier(9999)
