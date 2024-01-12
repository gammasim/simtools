#!/usr/bin/python3

import logging

import pytest

from simtools.model.mirrors import InvalidMirrorListFile, Mirrors

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def mirror_template_ecsv(io_handler):
    mirror_list_file = io_handler.get_input_data_file(
        file_name="mirror_list_CTA-N-LST1_v2019-03-31_rotated.ecsv",
        test=True,
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    mirror_template_ecsv = Mirrors(mirror_list_file)
    return mirror_template_ecsv


@pytest.fixture
def mirror_template_simtel(io_handler):
    mirror_list_file = io_handler.get_input_data_file(
        file_name="mirror_list_CTA-N-LST1_v2019-03-31_rotated_simtel.dat",
        test=True,
    )
    logger.info(f"Using mirror list with simtel format {mirror_list_file}")
    mirror_template_simtel = Mirrors(mirror_list_file)
    return mirror_template_simtel


@pytest.fixture
def mirror_table_template(io_handler):
    mirror_list_file = io_handler.get_input_data_file(
        file_name="mirror_list_CTA-N-LST1_v2019-03-31_rotated.ecsv",
        test=True,
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    mirrors = Mirrors(mirror_list_file)
    mirror_table_template = mirrors.mirror_table.copy()
    return mirror_table_template


def write_tmp_mirror_list(io_handler, tmp_test_directory, incomplete_mirror_table):
    with open(tmp_test_directory / "incomplete_mirror_table.ecsv", "w"):
        incomplete_mirror_table.write(
            f"{tmp_test_directory}/incomplete_mirror_table.ecsv",
            format="ascii.ecsv",
            overwrite=True,
        )
    mirror_list_file = io_handler.get_input_data_file(
        file_name=f"{tmp_test_directory}/incomplete_mirror_table.ecsv",
        test=True,
    )
    return mirror_list_file


def test_read_mirror_list_from_sim_telarray(io_handler, mirror_template_simtel):
    mirrors = mirror_template_simtel
    assert 198 == mirrors.number_of_mirrors
    assert 151.0 == pytest.approx(mirrors.mirror_diameter.value)
    assert 3 == mirrors.shape_type


def test_read_mirror_list_from_ecsv(io_handler, mirror_template_ecsv):
    mirrors = mirror_template_ecsv
    assert 198 == mirrors.number_of_mirrors
    assert 151.0 == pytest.approx(mirrors.mirror_diameter.value)
    assert 3 == mirrors.shape_type


def test_read_mirror_list_from_ecsv_missing_mirror_diameter(
    io_handler, tmp_test_directory, mirror_table_template, caplog
):
    incomplete_mirror_table = mirror_table_template
    incomplete_mirror_table.remove_column("mirror_diameter")
    mirror_list_file = write_tmp_mirror_list(
        io_handler, tmp_test_directory, incomplete_mirror_table
    )
    with pytest.raises(TypeError):
        Mirrors(mirror_list_file)
    with caplog.at_level(logging.DEBUG):
        Mirrors(
            mirror_list_file, parameters={"mirror_panel_diameter": {"Value": 150, "units": "cm"}}
        )
    assert "Take mirror_panel_diameter from parameters" in caplog.text


def test_read_mirror_list_from_ecsv_missing_focal_length(
    io_handler, tmp_test_directory, mirror_table_template, caplog
):
    incomplete_mirror_table = mirror_table_template
    incomplete_mirror_table.remove_column("focal_length")
    mirror_list_file = write_tmp_mirror_list(
        io_handler, tmp_test_directory, incomplete_mirror_table
    )
    with pytest.raises(TypeError):
        Mirrors(mirror_list_file)
    with caplog.at_level(logging.DEBUG):
        Mirrors(
            mirror_list_file, parameters={"mirror_focal_length": {"Value": 2000, "units": "cm"}}
        )
    assert "Take mirror_focal_length from parameters" in caplog.text


def test_read_mirror_list_from_ecsv_missing_shape_type(
    io_handler, tmp_test_directory, mirror_table_template, caplog
):
    incomplete_mirror_table = mirror_table_template
    incomplete_mirror_table.remove_column("shape_type")
    mirror_list_file = write_tmp_mirror_list(
        io_handler, tmp_test_directory, incomplete_mirror_table
    )
    with pytest.raises(TypeError):
        Mirrors(mirror_list_file)
    with caplog.at_level(logging.DEBUG):
        Mirrors(mirror_list_file, parameters={"mirror_panel_shape": {"Value": 3}})
    assert "Take shape_type from parameters" in caplog.text


def test_read_mirror_list_from_ecsv_empty(io_handler, tmp_test_directory, mirror_table_template):
    incomplete_mirror_table = mirror_table_template
    incomplete_mirror_table.remove_rows(slice(0, len(incomplete_mirror_table)))
    logger.info("Using empty mirror table")
    mirror_list_file = write_tmp_mirror_list(
        io_handler, tmp_test_directory, incomplete_mirror_table
    )
    with pytest.raises(InvalidMirrorListFile):
        Mirrors(mirror_list_file)


def test_read_mirror_list_from_ecsv_no_DB(io_handler):
    mirror_list_file = io_handler.get_input_data_file(
        file_name="MLTdata-preproduction.ecsv",
        test=True,
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    with pytest.raises(TypeError):
        Mirrors(mirror_list_file)


def assert_mirror_parameters(mirror_x, mirror_y, mirror_diameter, focal_length, shape_type):
    assert 1022.49 == mirror_x.value
    assert -462.0 == mirror_y.value
    assert 151.0 == mirror_diameter.value
    assert 2920.0 == focal_length.value
    assert 3 == shape_type.value


def test_get_single_mirror_parameters_ecsv(io_handler, mirror_template_ecsv):
    mirrors = mirror_template_ecsv
    assert_mirror_parameters(*mirrors.get_single_mirror_parameters(198))


def test_get_single_mirror_parameters_simtel(io_handler, mirror_template_simtel):
    mirrors = mirror_template_simtel
    assert_mirror_parameters(*mirrors.get_single_mirror_parameters(198))


def test_get_single_mirror_parameters_simtel_wrong_id(io_handler, mirror_template_simtel):
    logger.info("Wrong mirror id returns the first mirror table row")
    mirrors = mirror_template_simtel
    assert_mirror_parameters(*mirrors.get_single_mirror_parameters(9999))


def test_get_single_mirror_parameters_simtel_missing_column(io_handler, mirror_template_simtel):
    logger.info("Removing column mirror_x")
    mirrors = mirror_template_simtel
    mirrors.mirror_table.rename_column("mirror_x", "mirror_xa")
    (
        mirror_x,
        mirror_y,
        mirror_diameter,
        focal_length,
        shape_type,
    ) = mirrors.get_single_mirror_parameters(198)
    assert 0 == mirror_x
    assert 2920.0 == focal_length.value
