#!/usr/bin/python3

import logging

import pytest

from simtools.model.mirrors import InvalidMirrorListFileError, Mirrors

logger = logging.getLogger()


@pytest.fixture
def mirror_template_ecsv(io_handler):
    mirror_list_file = io_handler.get_test_data_file(
        file_name="mirror_list_CTA-N-LST1_v2019-03-31_rotated.ecsv",
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    return Mirrors(mirror_list_file)


@pytest.fixture
def mirror_template_simtel(io_handler):
    mirror_list_file = io_handler.get_test_data_file(
        file_name="mirror_list_CTA-N-LST1_v2019-03-31_rotated_simtel.dat",
    )
    logger.info(f"Using mirror list with simtel format {mirror_list_file}")
    return Mirrors(mirror_list_file)


@pytest.fixture
def mirror_table_template(io_handler):
    mirror_list_file = io_handler.get_test_data_file(
        file_name="mirror_list_CTA-N-LST1_v2019-03-31_rotated.ecsv",
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    mirrors = Mirrors(mirror_list_file)
    return mirrors.mirror_table.copy()


def write_tmp_mirror_list(io_handler, tmp_test_directory, incomplete_mirror_table):
    with open(tmp_test_directory / "incomplete_mirror_table.ecsv", "w"):
        incomplete_mirror_table.write(
            f"{tmp_test_directory}/incomplete_mirror_table.ecsv",
            format="ascii.ecsv",
            overwrite=True,
        )
    return io_handler.get_test_data_file(
        file_name=f"{tmp_test_directory}/incomplete_mirror_table.ecsv",
    )


def test_read_mirror_list_from_sim_telarray(io_handler, mirror_template_simtel, tmp_test_directory):
    mirrors = mirror_template_simtel
    assert 198 == mirrors.number_of_mirrors
    assert mirrors.mirror_diameter.value == pytest.approx(151.0)
    assert 3 == mirrors.shape_type

    # reduced table with less columns
    columns_to_write = ["mirror_x", "mirror_y", "mirror_diameter", "focal_length", "shape_type"]
    tmp_mirror_list = tmp_test_directory / "mirror_list_5columns.txt"
    mirrors.mirror_table[columns_to_write].write(tmp_mirror_list, format="ascii.no_header")

    red_mirrors = Mirrors(mirror_list_file=tmp_mirror_list)
    assert 198 == red_mirrors.number_of_mirrors
    assert red_mirrors.mirror_table["mirror_panel_id"][0] == 0
    assert red_mirrors.mirror_table["mirror_panel_id"][5] == 5
    assert "mirror_z" not in red_mirrors.mirror_table.columns


def test_read_mirror_list_from_ecsv(io_handler, mirror_template_ecsv):
    mirrors = mirror_template_ecsv
    assert 198 == mirrors.number_of_mirrors
    assert mirrors.mirror_diameter.value == pytest.approx(151.0)
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
            mirror_list_file, parameters={"mirror_panel_diameter": {"value": 150, "unit": "cm"}}
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
        Mirrors(mirror_list_file, parameters={"mirror_focal_length": {"value": 2000, "unit": "cm"}})
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
        Mirrors(mirror_list_file, parameters={"mirror_panel_shape": {"value": 3}})
    assert "Take shape_type from parameters" in caplog.text


def test_read_mirror_list_from_ecsv_empty(io_handler, tmp_test_directory, mirror_table_template):
    incomplete_mirror_table = mirror_table_template
    incomplete_mirror_table.remove_rows(slice(0, len(incomplete_mirror_table)))
    logger.info("Using empty mirror table")
    mirror_list_file = write_tmp_mirror_list(
        io_handler, tmp_test_directory, incomplete_mirror_table
    )
    with pytest.raises(InvalidMirrorListFileError):
        Mirrors(mirror_list_file)


def test_read_mirror_list_from_ecsv_no_db(io_handler):
    mirror_list_file = io_handler.get_test_data_file(
        file_name="MLTdata-preproduction.ecsv",
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    with pytest.raises(TypeError):
        Mirrors(mirror_list_file)


def assert_mirror_parameters(mirror_x, mirror_y, mirror_diameter, focal_length, shape_type):
    assert mirror_x.value == pytest.approx(1022.49)
    assert mirror_y.value == pytest.approx(-462.0)
    assert mirror_diameter.value == pytest.approx(151.0)
    assert focal_length.value == pytest.approx(2920.0)
    assert shape_type.value == 3


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
        _mirror_y,
        _mirror_diameter,
        focal_length,
        _shape_type,
    ) = mirrors.get_single_mirror_parameters(198)
    assert 0 == mirror_x
    assert focal_length.value == pytest.approx(2920.0)
