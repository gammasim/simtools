"""Tests for the derive-flasher-dc-pixel-map application."""

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from simtools.applications import derive_flasher_dc_pixel_map as app
from simtools.configuration.commandline_parser import CommandLineParser


def test_parser_defaults():
    parser = CommandLineParser()
    parser.add_argument_definitions(app._ARGUMENTS)
    args = parser.parse_args(["--simtel_files", "a.simtel.zst", "--telescope", "MSTN-design"])

    assert args.simtel_files == ["a.simtel.zst"]
    assert args.telescope == "MSTN-design"
    assert args.output_file == "flasher_dc_relative_pixel_map.ecsv"
    assert args.plot is False


def test_parser_requires_telescope():
    parser = CommandLineParser()
    parser.add_argument_definitions(app._ARGUMENTS)
    with pytest.raises(SystemExit):
        parser.parse_args(["--simtel_files", "a.simtel.zst"])


def test_make_pixel_map_normalizes_active_pixels_and_preserves_disabled_zeros():
    table = app._make_pixel_map(np.array([2.0, 4.0, 0.0, 100.0]), [True, True, True, False])

    np.testing.assert_allclose(table["relative_rate"], [1.0, 2.0, 0.0, 0.0])
    np.testing.assert_allclose(table["relative_rate_error"], np.sqrt([2.0, 4.0, 0.0, 0.0]) / 2.0)
    assert np.mean(table["relative_rate"][:3]) == pytest.approx(1.0)


def test_make_pixel_map_does_not_infer_disabled_pixels_from_zero_counts():
    table = app._make_pixel_map(np.array([0.0, 4.0, 8.0]), [True, True, True])

    np.testing.assert_allclose(table["relative_rate"], [0.0, 1.0, 2.0])


def test_make_pixel_map_rejects_wrong_mask_size():
    with pytest.raises(ValueError, match="active pixel mask must match"):
        app._make_pixel_map(np.array([1.0, 2.0]), [True])


def test_make_pixel_map_rejects_no_active_pixels():
    with pytest.raises(ValueError, match="no active pixels"):
        app._make_pixel_map(np.array([0.0, 0.0]), [False, False])


def _simtel_files(directory, names):
    files = [Path(directory) / name for name in names]
    for file in files:
        file.touch()
    return files


def test_main_writes_pixel_map(mocker, tmp_test_directory, run_context):
    a_file, b_file = _simtel_files(tmp_test_directory, ["a.simtel.zst", "b.simtel.zst"])
    args = {
        "simtel_files": [str(Path(tmp_test_directory) / "*.simtel.zst")],
        "telescope": "MSTN-design",
        "output_file": "map.ecsv",
        "plot": False,
    }
    context = run_context.replace(args=args)
    mocker.patch.object(type(app.APPLICATION), "start", return_value=context)
    mocker.patch.object(
        app,
        "sum_true_photoelectrons",
        return_value=(np.array([1.0, 3.0]), 7),
    )
    mocker.patch.object(app, "active_pixel_mask", return_value=np.array([True, False]))

    app.main()

    output = context.paths.get_output_directory() / "map.ecsv"
    table = Table.read(output, format="ascii.ecsv")
    assert table.meta["number_of_events"] == 7
    assert table.meta["telescope"] == "MSTN-design"
    assert table.meta["input_files"] == [str(a_file), str(b_file)]
    np.testing.assert_allclose(table["relative_rate"], [1.0, 0.0])


def test_main_plots_pixel_map(mocker, tmp_test_directory, run_context):
    (a_file,) = _simtel_files(tmp_test_directory, ["a.simtel.zst"])
    args = {
        "simtel_files": [str(a_file)],
        "telescope": "MSTN-design",
        "output_file": "map.ecsv",
        "plot": True,
    }
    context = run_context.replace(args=args)
    mocker.patch.object(type(app.APPLICATION), "start", return_value=context)
    mocker.patch.object(app, "sum_true_photoelectrons", return_value=(np.array([1.0, 3.0]), 2))
    mocker.patch.object(app, "active_pixel_mask", return_value=np.array([True, True]))
    camera = mocker.patch.object(app, "read_camera", return_value="camera")
    plot = mocker.patch.object(app, "plot_pixel_map")

    app.main()

    camera.assert_called_once_with(a_file, "MSTN-design")
    plot.assert_called_once()
    assert plot.call_args.args[1] == "camera"
    assert plot.call_args.args[2].name == "map"


def test_main_raises_without_matching_files(mocker, tmp_test_directory, run_context):
    args = {
        "simtel_files": [str(Path(tmp_test_directory) / "missing_*.simtel.zst")],
        "telescope": "MSTN-design",
        "output_file": "map.ecsv",
        "plot": False,
    }
    mocker.patch.object(type(app.APPLICATION), "start", return_value=run_context.replace(args=args))

    with pytest.raises(FileNotFoundError, match="No files found"):
        app.main()
