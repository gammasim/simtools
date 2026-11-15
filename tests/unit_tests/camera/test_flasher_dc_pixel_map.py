"""Tests for the flasher DC relative pixel map."""

from unittest import mock

import numpy as np
import pytest
from astropy.table import Table

from simtools.camera import flasher_dc_pixel_map as pixel_map

TELESCOPE = "MSTN-design"
TEL_ID = 3


def _simtel_file(events):
    """Return a patched SimTelFile yielding the given events and a telescope ID mapping."""
    simtel = mock.MagicMock()
    simtel.__enter__.return_value = simtel
    simtel.__iter__.side_effect = lambda: iter(events)
    simtel.header = {"tel_id": [TEL_ID]}
    return (
        mock.patch.object(pixel_map, "SimTelFile", return_value=simtel),
        mock.patch.object(
            pixel_map,
            "get_sim_telarray_telescope_id_to_telescope_name_mapping",
            return_value={TEL_ID: TELESCOPE},
        ),
    )


def _event(counts, tel_id=0):
    return {"photoelectrons": {tel_id: {"photoelectrons": np.asarray(counts)}}}


def test_relative_pixel_map_mean_and_errors():
    pe_sum = np.array([2.0, 4.0, 6.0, 8.0])
    table = pixel_map.relative_pixel_map(pe_sum)

    assert isinstance(table, Table)
    assert list(table.colnames) == ["pixel_id", "relative_rate", "relative_rate_error"]
    assert table["pixel_id"].dtype == np.int64
    assert table["relative_rate"].dtype == np.float64
    assert table["relative_rate_error"].dtype == np.float64
    assert np.mean(table["relative_rate"]) == pytest.approx(1.0)
    np.testing.assert_allclose(table["relative_rate"], pe_sum / 5.0)
    np.testing.assert_allclose(table["relative_rate_error"], np.sqrt(pe_sum) / 5.0)
    np.testing.assert_array_equal(table["pixel_id"], [0, 1, 2, 3])


def test_relative_pixel_map_zero_mean_raises():
    with pytest.raises(ValueError, match="Mean photo-electron count per pixel is zero"):
        pixel_map.relative_pixel_map(np.zeros(4))


def test_relative_pixel_map_empty_raises():
    with pytest.raises(ValueError, match="Mean photo-electron count per pixel is zero"):
        pixel_map.relative_pixel_map(np.array([]))


def test_write_and_read_roundtrip(tmp_test_directory):
    table = pixel_map.relative_pixel_map(np.array([1.0, 3.0]))
    output = f"{tmp_test_directory}/map.ecsv"

    pixel_map.write_pixel_map(
        table,
        output,
        number_of_events=10,
        telescope=TELESCOPE,
        input_files=["a.simtel.zst", "b.simtel.zst"],
    )
    read_back = Table.read(output, format="ascii.ecsv")

    assert read_back.meta["number_of_events"] == 10
    assert read_back.meta["telescope"] == TELESCOPE
    assert read_back.meta["input_files"] == ["a.simtel.zst", "b.simtel.zst"]
    np.testing.assert_allclose(read_back["relative_rate"], table["relative_rate"])
    np.testing.assert_allclose(read_back["relative_rate_error"], table["relative_rate_error"])
    np.testing.assert_array_equal(read_back["pixel_id"], table["pixel_id"])


def test_write_pixel_map_without_metadata(tmp_test_directory):
    output = f"{tmp_test_directory}/plain.ecsv"
    pixel_map.write_pixel_map(pixel_map.relative_pixel_map(np.array([1.0, 1.0])), output)

    read_back = Table.read(output, format="ascii.ecsv")
    assert "number_of_events" not in read_back.meta
    assert "input_files" not in read_back.meta


def test_sum_true_photoelectrons_sums_over_events():
    events = [_event([1, 0, 2]), {"photoelectrons": {}}, _event([0, 3, 1])]
    file_patch, mapping_patch = _simtel_file(events)
    with file_patch, mapping_patch:
        pe_sum, number_of_events = pixel_map.sum_true_photoelectrons(["run.simtel.zst"], TELESCOPE)

    # Provenance records only events that contributed a PE record for this telescope.
    assert number_of_events == 2
    np.testing.assert_allclose(pe_sum, [1.0, 3.0, 3.0])


def test_photoelectron_telescope_id_uses_run_header_order():
    # Run-header telescope IDs can be sparse; PE IDs use their array positions.
    assert pixel_map._photoelectron_telescope_id([1, 3, 7], 7, "run.simtel.zst") == 2


def test_photoelectron_telescope_id_missing_header_id_raises():
    with pytest.raises(ValueError, match="not found in simtel run header"):
        pixel_map._photoelectron_telescope_id([1, 3], 7, "run.simtel.zst")


def test_sum_true_photoelectrons_sums_over_files():
    file_patch, mapping_patch = _simtel_file([_event([1, 2])])
    with file_patch, mapping_patch:
        pe_sum, number_of_events = pixel_map.sum_true_photoelectrons(
            ["a.simtel.zst", "b.simtel.zst"], TELESCOPE
        )

    assert number_of_events == 2
    np.testing.assert_allclose(pe_sum, [2.0, 4.0])


def test_sum_true_photoelectrons_ignores_other_telescopes():
    events = [_event([5, 5], tel_id=1), _event([1, 1])]
    file_patch, mapping_patch = _simtel_file(events)
    with file_patch, mapping_patch:
        pe_sum, _ = pixel_map.sum_true_photoelectrons(["run.simtel.zst"], TELESCOPE)

    np.testing.assert_allclose(pe_sum, [1.0, 1.0])


def test_sum_true_photoelectrons_without_stored_photoelectrons_raises():
    file_patch, mapping_patch = _simtel_file([{"photoelectrons": {}}, {}])
    with file_patch, mapping_patch:
        with pytest.raises(ValueError, match="store_photoelectrons"):
            pixel_map.sum_true_photoelectrons(["run.simtel.zst"], TELESCOPE)


def test_sum_true_photoelectrons_inconsistent_pixel_count_raises():
    file_patch, mapping_patch = _simtel_file([_event([1, 2]), _event([1, 2, 3])])
    with file_patch, mapping_patch:
        with pytest.raises(ValueError, match="Inconsistent number of pixels"):
            pixel_map.sum_true_photoelectrons(["run.simtel.zst"], TELESCOPE)


def test_sum_true_photoelectrons_unknown_telescope_raises():
    with mock.patch.object(
        pixel_map,
        "get_sim_telarray_telescope_id_to_telescope_name_mapping",
        return_value={TEL_ID: "LSTN-01"},
    ):
        with pytest.raises(ValueError, match="not found in file"):
            pixel_map.sum_true_photoelectrons(["run.simtel.zst"], TELESCOPE)


def test_plot_pixel_map_saves_figure(tmp_test_directory):
    table = pixel_map.relative_pixel_map(np.array([1.0, 3.0]))
    camera = mock.MagicMock()
    fig = mock.MagicMock()
    with (
        mock.patch.object(pixel_map, "plot_pixel_layout_with_image", return_value=fig) as layout,
        mock.patch.object(pixel_map, "save_figure") as save,
    ):
        pixel_map.plot_pixel_map(table, camera, f"{tmp_test_directory}/map")

    layout.assert_called_once()
    np.testing.assert_allclose(layout.call_args.kwargs["image"], table["relative_rate"])
    save.assert_called_once()
    assert save.call_args.args[0] is fig
