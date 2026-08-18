#!/usr/bin/python3

import pytest

from simtools.io import legacy_data_handler


@pytest.fixture
def test_spe_file(tmp_test_directory):
    test_file = tmp_test_directory / "single_pe.csv"
    test_file.write_text("0.1,1.0\n0.2,2.0\n", encoding="utf-8")
    return test_file


def test_read_legacy_data_file(test_spe_file):
    table = legacy_data_handler.read_legacy_data_as_table(test_spe_file, "legacy_lst_single_pe")
    assert table.colnames == ["amplitude", "response"]

    with pytest.raises(ValueError, match="Unsupported legacy data file type: not_a_file_type"):
        legacy_data_handler.read_legacy_data_as_table(test_spe_file, "not_a_file_type")
