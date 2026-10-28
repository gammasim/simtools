#!/usr/bin/python3

import pytest

from simtools.visualization import legend_handlers as leg_h


class MockHandleBox:
    """Mock handlebox for testing legend handlers."""

    def __init__(self, xdescent=5, ydescent=10, width=20, height=30):
        self.xdescent = xdescent
        self.ydescent = ydescent
        self.width = width
        self.height = height
        self.artist = None

    def get_transform(self):
        return None

    def add_artist(self, artist):
        self.artist = artist


def test_calculate_center():
    handlebox = MockHandleBox(xdescent=10, ydescent=20, width=30, height=40)

    x0, y0 = leg_h.calculate_center(handlebox, width_factor=3, height_factor=4)
    assert x0 == pytest.approx(10 + 30 / 3)
    assert y0 == pytest.approx(20 + 40 / 4)

    x0, y0 = leg_h.calculate_center(handlebox, width_factor=2, height_factor=2)
    assert x0 == pytest.approx(10 + 30 / 2)
    assert y0 == pytest.approx(20 + 40 / 2)


def test_base_hex_pixel_handler_create_hex_patch():
    handlebox = MockHandleBox()
    facecolor = (0.5, 0.5, 0.5, 1)
    edgecolor = (0, 0, 0, 1)

    patch = leg_h._BaseHexPixelHandler._create_hex_patch(handlebox, facecolor, edgecolor)

    assert patch is not None
    assert patch.get_facecolor() == facecolor
    assert patch.get_edgecolor() == edgecolor
    assert patch.numvertices == 6
    assert handlebox.artist == patch


def test_base_square_pixel_handler_create_square_patch():
    handlebox = MockHandleBox()
    facecolor = (0.3, 0.3, 0.3, 1)
    edgecolor = (0, 0, 0, 1)

    patch = leg_h._BaseSquarePixelHandler._create_square_patch(handlebox, facecolor, edgecolor)

    assert patch is not None
    assert patch.get_facecolor() == facecolor
    assert patch.get_edgecolor() == edgecolor
    assert patch.get_width() == handlebox.height
    assert patch.get_height() == handlebox.height
    assert handlebox.artist == patch


def test_telescope_handler():
    for telescope_type, config in leg_h.TELESCOPE_CONFIG.items():
        handler = leg_h.BaseLegendHandler(telescope_type)
        assert handler.telescope_type == telescope_type
        assert handler.config == config

        handlebox = MockHandleBox()
        patch = handler.legend_artist(None, None, None, handlebox)

        assert patch is not None
        assert handlebox.artist == patch

        if config["shape"] == "circle":
            assert isinstance(patch, leg_h.mpatches.Circle)
        elif config["shape"] == "square":
            assert isinstance(patch, leg_h.mpatches.Rectangle)
        elif config["shape"] == "hexagon":
            assert isinstance(patch, leg_h.mpatches.RegularPolygon)


def test_various_telescope_handlers():
    handlebox = MockHandleBox()

    lst_handler = leg_h.LSTHandler()
    assert lst_handler.telescope_type == "LST"
    patch = lst_handler.legend_artist(None, None, None, handlebox)
    assert patch is not None
    assert leg_h.MSTHandler().telescope_type == "MST"
    assert leg_h.SSTHandler().telescope_type == "SST"
    assert leg_h.SCTHandler().telescope_type == "SCT"
    assert leg_h.VERITASHandler().telescope_type == "VERITAS"
    assert leg_h.HESSHandler().telescope_type == "HESS"
    assert leg_h.MAGICHandler().telescope_type == "MAGIC"


def test_base_legend_handler_raises_on_unknown_shape():
    class DummyHandleBox:
        xdescent = 0
        ydescent = 0
        width = 10
        height = 10

        def get_transform(self):
            return None

        def add_artist(self, artist):
            self.artist = artist

    handler = leg_h.BaseLegendHandler("LST")
    handler.config["shape"] = "unknown_shape"
    handlebox = DummyHandleBox()
    with pytest.raises(ValueError, match="Unknown shape: unknown_shape"):
        handler.legend_artist(None, None, None, handlebox)
