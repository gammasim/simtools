#!/usr/bin/python3

import filecmp
import logging

import pytest
from astropy import units as u

from simtools.model.model_parameter import InvalidModelParameter
from simtools.model.telescope_model import TelescopeModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def lst_config_file():
    """Return the path to test config file for LSTN-01"""
    return "tests/resources/CTA-North-LSTN-01-Released_test-telescope-model.cfg"


@pytest.fixture
def telescope_model_from_config_file(io_handler, lst_config_file):
    label = "test-telescope-model"
    tel_model = TelescopeModel.from_config_file(
        site="North",
        telescope_model_name="LSTN-01",
        label=label,
        config_file_name=lst_config_file,
    )
    return tel_model


def test_get_parameter_dict(telescope_model_lst):
    tel_model = telescope_model_lst
    assert isinstance(tel_model.get_parameter_dict("num_gains"), dict)
    assert isinstance(tel_model.get_parameter_dict("num_gains")["value"], int)
    assert isinstance(tel_model.get_parameter_dict("telescope_axis_height")["value"], float)
    assert tel_model.get_parameter_dict("telescope_axis_height")["unit"] == "m"

    with pytest.raises(InvalidModelParameter):
        tel_model.get_parameter_dict("not_a_parameter")


def test_get_parameter_value(telescope_model_lst):
    tel_model = telescope_model_lst
    assert isinstance(tel_model.get_parameter_value("num_gains"), int)

    _par_dict_value_missing = {"unit": "m", "type": "float"}
    with pytest.raises(KeyError):
        tel_model.get_parameter_value("num_gains", parameter_dict=_par_dict_value_missing)


def test_get_parameter_value_with_unit(telescope_model_lst):
    tel_model = telescope_model_lst

    assert isinstance(tel_model.get_parameter_value_with_unit("fadc_mhz"), u.Quantity)
    assert not isinstance(tel_model.get_parameter_value_with_unit("num_gains"), u.Quantity)


def test_get_parameter_value_as_list(telescope_model_lst):

    tel_model = telescope_model_lst
    t_1 = tel_model.get_parameter_value_as_list("telescope_transmission")
    assert isinstance(t_1, list)
    assert len(t_1) == 6

    t_dont_extend = tel_model.get_parameter_value_as_list("telescope_transmission", 12)
    assert len(t_dont_extend) == 6

    t_single_float = tel_model.get_parameter_value_as_list("camera_transmission")
    assert len(t_single_float) == 1
    t_four_floats = tel_model.get_parameter_value_as_list("camera_transmission", 4, 1.0)
    assert len(t_four_floats) == 4
    assert pytest.approx(t_four_floats[3]) == 1.0


def test_handling_parameters(telescope_model_lst):
    tel_model = telescope_model_lst

    logger.info(
        "Old mirror_reflection_random_angle: "
        f"{tel_model.get_parameter_value('mirror_reflection_random_angle')}"
    )
    logger.info("Changing mirror_reflection_random_angle")
    new_mrra = "0.0080 0 0"
    tel_model.change_parameter("mirror_reflection_random_angle", new_mrra)

    assert new_mrra == tel_model.get_parameter_value("mirror_reflection_random_angle")

    logging.info("Adding new_parameter")
    new_par = "23"
    tel_model.add_parameter("new_parameter", new_par)

    assert new_par == tel_model.get_parameter_value("new_parameter")

    with pytest.raises(InvalidModelParameter):
        tel_model.get_parameter_dict("bla_bla")


def test_change_parameter(telescope_model_lst):
    tel_model = telescope_model_lst

    logger.info(f"Old camera_pixels:{tel_model.get_parameter_value('camera_pixels')}")
    tel_model.change_parameter("camera_pixels", 9999)
    assert 9999 == tel_model.get_parameter_value("camera_pixels")

    with pytest.raises(ValueError):
        logger.info("Testing changing camera_pixels to a float (now allowed)")
        tel_model.change_parameter("camera_pixels", 9999.9)

    with pytest.raises(ValueError):
        logger.info("Testing changing camera_pixels to a nonsense string")
        tel_model.change_parameter("camera_pixels", "bla_bla")

    logger.info(f"Old camera_pixels:{tel_model.get_parameter_value('mirror_focal_length')}")
    tel_model.change_parameter("mirror_focal_length", 55.0)
    assert pytest.approx(55.0) == tel_model.get_parameter_value("mirror_focal_length")
    tel_model.change_parameter("mirror_focal_length", 55)
    assert pytest.approx(55.0) == tel_model.get_parameter_value("mirror_focal_length")

    with pytest.raises(ValueError):
        logger.info("Testing changing mirror_focal_length to a nonsense string")
        tel_model.change_parameter("mirror_focal_length", "bla_bla")


def test_flen_type(telescope_model_lst):
    tel_model = telescope_model_lst
    flen_info = tel_model.get_parameter_dict("focal_length")
    logger.info(f"Focal Length = {flen_info['value']}, type = {flen_info['type']}")

    assert isinstance(flen_info["value"], float)


def test_cfg_file(telescope_model_from_config_file, lst_config_file):
    tel_model = telescope_model_from_config_file

    tel_model.export_config_file()

    logger.info(f"Config file (original): {lst_config_file}")
    logger.info(f"Config file (new): {tel_model.get_config_file()}")

    assert filecmp.cmp(lst_config_file, tel_model.get_config_file())

    cfg_file = tel_model.get_config_file()
    tel = TelescopeModel.from_config_file(
        site="south",
        telescope_model_name="ssts-design",
        label="test-sst",
        config_file_name=cfg_file,
    )
    tel.export_config_file()
    logger.info(f"Config file (sst): {tel.get_config_file()}")
    # TODO: testing that file can be written and that it is different,
    #       but not the file has the
    #       correct contents
    assert False is filecmp.cmp(lst_config_file, tel.get_config_file())


def test_updating_export_model_files(db_config, io_handler):
    """
    It was found in derive_mirror_rnda_angle that the DB was being
    accessed each time the model was changed, because the model
    files were being re-exported. A flag called _is_exported_model_files_up_to_date
    was added to prevent this behavior. This test is meant to assure
    it is working properly.
    """

    # We need a brand new telescope_model to avoid interference
    tel = TelescopeModel(
        site="North",
        telescope_model_name="LSTN-01",
        model_version="prod4",
        label="test-telescope-model-2",
        mongo_db_config=db_config,
    )

    logger.debug(
        "tel._is_exported_model_files should be False because export_config_file"
        " was not called yet."
    )
    assert False is tel._is_exported_model_files_up_to_date

    # Exporting config file
    tel.export_config_file()
    logger.debug(
        "tel._is_exported_model_files should be True because export_config_file" " was called."
    )
    assert tel._is_exported_model_files_up_to_date

    # Changing a non-file parameter
    logger.info("Changing a parameter that IS NOT a file - mirror_reflection_random_angle")
    tel.change_parameter("mirror_reflection_random_angle", "0.0080 0 0")
    logger.debug(
        "tel._is_exported_model_files should still be True because the changed "
        "parameter was not a file"
    )
    assert tel._is_exported_model_files_up_to_date

    # Testing the DB connection
    logger.info("DB should NOT be read next.")
    tel.export_config_file()

    # Changing a parameter that is a file
    logger.debug("Changing a parameter that IS a file - camera_config_file")
    tel.change_parameter("camera_config_file", tel.get_parameter_value("camera_config_file"))
    logger.debug(
        "tel._is_exported_model_files should be False because a parameter that "
        "is a file was changed."
    )
    assert False is tel._is_exported_model_files_up_to_date


def test_export_derived_files(io_handler, db_config):
    tel_model = TelescopeModel(
        site="North",
        telescope_model_name="LSTN-01",
        model_version="Prod5",
        mongo_db_config=db_config,
        label="test-telescope-model-lst",
    )

    _ = tel_model.derived
    assert (
        tel_model.get_derived_directory()
        .joinpath("ray-tracing-North-LST-1-d10.0-za20.0_validate_optics.ecsv")
        .exists()
    )


def test_load_reference_data(telescope_model_lst):
    tel_model = telescope_model_lst

    assert tel_model.reference_data["nsb_reference_value"]["value"] == pytest.approx(0.24)


def test_get_reference_data_value(telescope_model_lst):
    tel_model = telescope_model_lst

    assert tel_model.get_reference_data_value("nsb_reference_value") == pytest.approx(0.24)

    with pytest.raises(KeyError):
        tel_model.get_reference_data_value("bla_bla")
