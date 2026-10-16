"""Test camera trigger selection arguments."""

import pytest

from simtools.applications.validate_camera_fov import _MAX_PIXEL_ID, APPLICATION

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
