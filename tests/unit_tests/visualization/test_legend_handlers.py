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
