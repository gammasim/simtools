#!/usr/bin/python3

import logging
from pathlib import Path

import pytest

from simtools.constants import SIM_TELARRAY_INCLUDE_FILENAME_MAX_LENGTH
from simtools.utils import names

logging.getLogger().setLevel(logging.DEBUG)

ecsv_suffix = ".ecsv"


@pytest.fixture
def invalid_name():
    return "Invalid name"


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


def test_get_site_from_array_element_name(invalid_name):
    assert "North" == names.get_site_from_array_element_name("MSTN")
    assert "North" == names.get_site_from_array_element_name("MSTN-05")
    assert "South" == names.get_site_from_array_element_name("MSTS-05")
    with pytest.raises(ValueError, match=rf"^{invalid_name}"):
        names.get_site_from_array_element_name("LSTW")
    assert ["North", "South"] == names.get_site_from_array_element_name("MSTx")
    assert "North" == names.get_site_from_array_element_name("OBS-North")
    assert "South" == names.get_site_from_array_element_name("OBS-South")
    assert "South" == names.get_site_from_array_element_name("South")


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


@pytest.mark.parametrize(
    ("stem", "suffix", "site", "telescope", "zenith", "azimuth", "label", "expected"),
    [
        (
            "camera_efficiency_table",
            ecsv_suffix,
            "South",
            "LSTS-01",
            20,
            180,
            "test",
            "camera_efficiency_table_South_LSTS-01_za20.0deg_azm180deg_test.ecsv",
        ),
        (
            "camera_efficiency",
            ".dat",
            "South",
            "LSTS-01",
            20,
            180,
            "test",
            "camera_efficiency_South_LSTS-01_za20.0deg_azm180deg_test.dat",
        ),
        (
            "camera_efficiency",
            ".log",
            "South",
            "LSTS-01",
            20,
            180,
            "test",
            "camera_efficiency_South_LSTS-01_za20.0deg_azm180deg_test.log",
        ),
        (
            "camera_efficiency_table",
            ecsv_suffix,
            "North",
            "MSTN",
            40,
            0,
            "test",
            "camera_efficiency_table_North_MSTN_za40.0deg_azm000deg_test.ecsv",
        ),
        (
            "camera_efficiency",
            ".dat",
            "North",
            "MSTN",
            40,
            0,
            "test",
            "camera_efficiency_North_MSTN_za40.0deg_azm000deg_test.dat",
        ),
        (
            "camera_efficiency",
            ".log",
            "North",
            "MSTN",
            40,
            0,
            "test",
            "camera_efficiency_North_MSTN_za40.0deg_azm000deg_test.log",
        ),
        (
            "camera_efficiency_table",
            ecsv_suffix,
            "South",
            "LSTS-01",
            20,
            180,
            None,
            "camera_efficiency_table_South_LSTS-01_za20.0deg_azm180deg.ecsv",
        ),
    ],
    ids=[
        "south-table",
        "south-dat",
        "south-log",
        "north-table",
        "north-dat",
        "north-log",
        "without-label",
    ],
)
def test_generate_file_name_camera_efficiency(
    stem, suffix, site, telescope, zenith, azimuth, label, expected
):
    assert (
        names.generate_file_name(stem, suffix, site, telescope, zenith, azimuth, label=label)
        == expected
    )


def test_simtel_config_file_name():
    assert (
        names.sim_telarray_config_file_name("South", telescope_model_name="LSTS-01")
        == "CTAO-LSTS-01.cfg"
    )
    assert (
        names.sim_telarray_config_file_name(array_name="4LSTs", site="South")
        == "CTAO-South-4LSTs.cfg"
    )


def test_simtel_config_file_name_too_long():
    too_long_telescope_name = "A" * SIM_TELARRAY_INCLUDE_FILENAME_MAX_LENGTH

    with pytest.raises(ValueError, match="exceeds the maximum length"):
        names.sim_telarray_config_file_name("South", telescope_model_name=too_long_telescope_name)


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


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "file_type": "photons",
                "suffix": ".lis",
                "off_axis_angle": 2.5,
                "mirror_number": 3,
                "label": "instance1",
            },
            "photons_South_LSTS-01_d10.5km_za45.0deg_off2.500deg_mirror3_instance1.lis",
        ),
        (
            {"file_type": "log", "suffix": ".log", "off_axis_angle": 2.5},
            "log_South_LSTS-01_d10.5km_za45.0deg_off2.500deg.log",
        ),
        (
            {"file_type": "ray_tracing", "suffix": ecsv_suffix, "label": "instance1"},
            "ray_tracing_South_LSTS-01_d10.5km_za45.0deg_instance1.ecsv",
        ),
        (
            {"file_type": "ray_tracing", "suffix": ecsv_suffix, "label": None},
            "ray_tracing_South_LSTS-01_d10.5km_za45.0deg.ecsv",
        ),
        (
            {
                "file_type": "ray_tracing",
                "suffix": ".pdf",
                "extra_label": "d80_cm",
                "label": "instance1",
            },
            "ray_tracing_South_LSTS-01_d10.5km_za45.0deg_instance1_d80_cm.pdf",
        ),
        (
            {"file_type": "ray_tracing", "suffix": ".pdf", "extra_label": "d80_cm", "label": None},
            "ray_tracing_South_LSTS-01_d10.5km_za45.0deg_d80_cm.pdf",
        ),
    ],
    ids=[
        "photons",
        "log",
        "ray-tracing-labeled",
        "ray-tracing",
        "extra-label",
        "extra-label-no-label",
    ],
)
def test_generate_file_name_ray_tracing(kwargs, expected):
    common = {
        "site": "South",
        "telescope_model_name": "LSTS-01",
        "source_distance": 10.5,
        "zenith_angle": 45.0,
    }
    assert names.generate_file_name(**common, **kwargs) == expected


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
        == "longitude"
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
            "reference_point_longitude", software_name=sim_telarray
        )
        == "longitude"
    )
    assert (
        names.get_simulation_software_meta_parameter_mode(
            "reference_point_longitude", software_name=sim_telarray
        )
        == "set"
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


def test_normalize_array_element_identifier():
    assert names.normalize_array_element_identifier(1) == "LSTN-01"
    assert names.normalize_array_element_identifier("1") == "LSTN-01"
    assert names.normalize_array_element_identifier("MSTN-15") == "MSTN-15"
    assert names.normalize_array_element_identifier("9999") == "9999"


def test_normalize_array_element_identifier_container():
    assert names.normalize_array_element_identifier_container([1, "MSTN-15", "9999"]) == [
        "LSTN-01",
        "MSTN-15",
        "9999",
    ]
    assert names.normalize_array_element_identifier_container("[1, 12]") == ["LSTN-01", "MSTN-08"]
    assert names.normalize_array_element_identifier_container(None) == []

    with pytest.raises(ValueError, match="Invalid JSON list string"):
        names.normalize_array_element_identifier_container("[1, 12")
