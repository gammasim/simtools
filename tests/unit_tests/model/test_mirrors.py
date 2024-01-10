#!/usr/bin/python3

import logging

import pytest
from astropy.utils.diff import report_diff_values

from simtools.model.mirrors import InvalidMirrorListFile, Mirrors

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_read_mirror_list_from_sim_telarray(db, io_handler):
    mirror_list_file = io_handler.get_input_data_file(
        file_name="mirror_list_CTA-N-LST1_v2019-03-31_rotated_simtel.dat",
        test=True,
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    mirrors = Mirrors(mirror_list_file)
    assert 198 == mirrors.number_of_mirrors
    assert 151.0 == pytest.approx(mirrors.mirror_diameter.value)
    assert 3 == mirrors.shape_type


def test_read_mirror_list_from_ecsv(io_handler, tmp_test_directory, caplog):
    mirror_list_file = io_handler.get_input_data_file(
        file_name="mirror_list_CTA-N-LST1_v2019-03-31_rotated.ecsv",
        test=True,
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    mirrors = Mirrors(mirror_list_file)
    assert 198 == mirrors.number_of_mirrors
    assert 151.0 == pytest.approx(mirrors.mirror_diameter.value)
    assert 3 == mirrors.shape_type
    temp_mirror_table = mirrors.mirror_table.copy()
    incomplete_mirror_table = temp_mirror_table.copy()
    incomplete_mirror_table.remove_column("mirror_diameter")
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
    with pytest.raises(TypeError):
        mirrors = Mirrors(mirror_list_file)
    with caplog.at_level(logging.DEBUG):
        mirrors = Mirrors(
            mirror_list_file, parameters={"mirror_panel_diameter": {"Value": 150, "units": "cm"}}
        )
    assert "Take mirror_panel_diameter from parameters" in caplog.text
    incomplete_mirror_table = temp_mirror_table.copy()
    incomplete_mirror_table.remove_column("focal_length")
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
    with pytest.raises(TypeError):
        mirrors = Mirrors(mirror_list_file)
    with caplog.at_level(logging.DEBUG):
        mirrors = Mirrors(
            mirror_list_file, parameters={"mirror_focal_length": {"Value": 2000, "units": "cm"}}
        )
    assert "Take mirror_focal_length from parameters" in caplog.text
    incomplete_mirror_table = temp_mirror_table.copy()
    incomplete_mirror_table.remove_column("shape_type")
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
    with pytest.raises(TypeError):
        mirrors = Mirrors(mirror_list_file)
    with caplog.at_level(logging.DEBUG):
        mirrors = Mirrors(mirror_list_file, parameters={"mirror_panel_shape": {"Value": 3}})
    assert "Take shape_type from parameters" in caplog.text

    incomplete_mirror_table = temp_mirror_table.copy()
    incomplete_mirror_table.remove_rows(slice(0, len(incomplete_mirror_table)))
    logger.info("Using empty mirror table")
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
    with pytest.raises(InvalidMirrorListFile):
        mirrors = Mirrors(mirror_list_file)

    mirror_list_file = io_handler.get_input_data_file(
        file_name="MLTdata-preproduction.ecsv",
        test=True,
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    with pytest.raises(TypeError):
        mirrors = Mirrors(mirror_list_file)


def test_get_mirror_table(io_handler):
    mirror_list_file = io_handler.get_input_data_file(
        file_name="mirror_list_CTA-N-LST1_v2019-03-31_rotated.ecsv",
        test=True,
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    mirrors = Mirrors(mirror_list_file)

    report = report_diff_values(mirrors.mirror_table, mirrors.get_mirror_table())
    assert report is True


def test_get_single_mirror_parameters(io_handler):
    mirror_list_file = io_handler.get_input_data_file(
        file_name="mirror_list_CTA-N-LST1_v2019-03-31_rotated.ecsv",
        test=True,
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    mirrors = Mirrors(mirror_list_file)
    (
        mirror_x,
        mirror_y,
        mirror_diameter,
        focal_length,
        shape_type,
    ) = mirrors.get_single_mirror_parameters(198)

    def assert_mirror_parameters():
        assert 1022.49 == mirror_x.value
        assert -462.0 == mirror_y.value
        assert 151.0 == mirror_diameter.value
        assert 2920.0 == focal_length.value
        assert 3 == shape_type.value

    assert_mirror_parameters()

    mirror_list_file = io_handler.get_input_data_file(
        file_name="mirror_list_CTA-N-LST1_v2019-03-31_rotated_simtel.dat",
        test=True,
    )
    logger.info(f"Using mirror list with simtel format {mirror_list_file}")
    mirrors = Mirrors(mirror_list_file)
    (
        mirror_x,
        mirror_y,
        mirror_diameter,
        focal_length,
        shape_type,
    ) = mirrors.get_single_mirror_parameters(198)
    assert_mirror_parameters()

    logger.info("Wrong mirror id returns the first mirror table row")
    mirrors = Mirrors(mirror_list_file)
    (
        mirror_x,
        mirror_y,
        mirror_diameter,
        focal_length,
        shape_type,
    ) = mirrors.get_single_mirror_parameters(9999)
    assert_mirror_parameters()

    logger.info("Removing column mirror_x")
    mirrors = Mirrors(mirror_list_file)
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
