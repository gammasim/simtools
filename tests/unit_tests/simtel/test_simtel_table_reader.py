"""Tests for the explicit legacy sim_telarray import adapter."""

from unittest import mock

import pytest

import simtools.simtel.simtel_table_reader as simtel_table_reader


def test_read_simtel_data(simtel_spe_test_file):
    """The importer extracts numeric rows and historical headers."""
    data, meta, n_columns, n_dim = simtel_table_reader._read_simtel_data(simtel_spe_test_file)

    assert data
    assert "Norm_spe processing" in meta
    assert n_columns == 3
    assert n_dim is None


@mock.patch("simtools.io.ascii_handler.read_file_encoded_in_utf_or_latin")
def test_read_simtel_data_rpol(mock_read_file):
    """The importer identifies the RPOL axis declaration."""
    mock_read_file.return_value = [
        "#@RPOL@[ANGLE=] 2",
        "ANGLE= 0 20",
        "300 0.8 0.7",
    ]

    rows, metadata, n_columns, n_dim = simtel_table_reader._read_simtel_data("test_file")

    assert rows == [[300.0, 0.8, 0.7]]
    assert metadata == ""
    assert n_columns == 3
    assert n_dim == ["0", "20"]


def test_read_simtel_table_uses_explicit_parameter_parser(tmp_test_directory):
    """The historical importer still supports conversion of plain tables."""
    path = tmp_test_directory / "nsb.lis"
    path.write_text("300 2.0 6.0\n400 3.0 7.0\n", encoding="utf-8")

    table = simtel_table_reader.read_simtel_table("nsb_reference_spectrum", path)

    assert table.colnames == ["wavelength", "fnu", "differential photon rate"]
    assert table["differential photon rate"][0] == pytest.approx(6.0)


def test_read_simtel_table_rejects_unknown_parameter(simtel_spe_test_file):
    """Unknown legacy table formats fail rather than guessing a representation."""
    with pytest.raises(ValueError, match="Unsupported parameter"):
        simtel_table_reader.read_simtel_table("not_a_parameter", simtel_spe_test_file)
