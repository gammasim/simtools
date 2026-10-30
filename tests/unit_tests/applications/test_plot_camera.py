"""Test camera trigger selection arguments."""

import pytest

from simtools.applications.plot_camera import _MAX_PIXEL_ID, APPLICATION

TRIGGER_PIXEL_OPTION = "--trigger_pixel"


@pytest.mark.parametrize("pixel", [0, 75, _MAX_PIXEL_ID])
def test_trigger_pixel_argument(pixel):
    args = APPLICATION.build_parser().parse_args([TRIGGER_PIXEL_OPTION, str(pixel)])
    assert args.trigger_pixel == pixel
    assert args.trigger_patches is None


@pytest.mark.parametrize("value", ["-1", "abc", str(_MAX_PIXEL_ID + 1)])
def test_trigger_pixel_argument_invalid(value, capsys):
    with pytest.raises(SystemExit):
        APPLICATION.build_parser().parse_args([TRIGGER_PIXEL_OPTION, value])
    assert TRIGGER_PIXEL_OPTION in capsys.readouterr().err


def test_trigger_selection_mutually_exclusive(capsys):
    with pytest.raises(SystemExit):
        APPLICATION.build_parser().parse_args(
            [TRIGGER_PIXEL_OPTION, "75", "--trigger_patches", "1", "2"]
        )
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("patch_ids", "pixel", "expected"),
    [
        (None, None, "pixel_layout"),
        ([0, 1], None, "selected"),
        (None, 0, "selected"),
        ([0], 0, None),
    ],
)
def test_main_selects_plot_path(patch_ids, pixel, expected, mocker):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from simtools.applications import plot_camera

    camera = MagicMock()
    camera.pixels = {"pix_id": [0, 1]}
    camera.calc_fov.return_value = (4.5, 120.0)
    tel_model = MagicMock(camera=camera)
    tel_model.name = "LSTN-01"
    tel_model.get_telescope_effective_focal_length.return_value = 2800.0
    app_context = SimpleNamespace(
        args={
            "site": "North",
            "telescope": "LSTN-01",
            "model_version": "5.0.0",
            "camera_in_sky_coor": False,
            "print_pixels_id": 10,
            "trigger_patches": patch_ids,
            "trigger_pixel": pixel,
        },
        model_reader=MagicMock(),
        paths=MagicMock(),
        logger=MagicMock(),
    )
    mocker.patch.object(plot_camera, "APPLICATION").start.return_value = app_context
    mocker.patch.object(plot_camera, "TelescopeModel", return_value=tel_model)
    mocker.patch.object(plot_camera, "trigger_patches_for_pixel", return_value=[0])
    mock_layout = mocker.patch.object(plot_camera, "plot_pixel_layout")
    mock_read = mocker.patch.object(plot_camera, "read_selected_trigger_patches")
    mock_selected = mocker.patch.object(plot_camera, "plot_selected_trigger_patches")
    mock_save = mocker.patch.object(plot_camera, "save_figure")

    if expected is None:
        with pytest.raises(ValueError, match="not both"):
            plot_camera.main()
        mock_save.assert_not_called()
        return
    plot_camera.main()
    if expected == "pixel_layout":
        mock_layout.assert_called_once_with(camera, False, 10)
        mock_read.assert_not_called()
    else:
        mock_read.assert_called_once()
        mock_selected.assert_called_once()
        mock_layout.assert_not_called()
    mock_save.assert_called_once()
