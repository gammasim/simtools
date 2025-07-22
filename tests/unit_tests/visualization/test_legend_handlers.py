#!/usr/bin/python3

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


def test_objects():
    object_list = [
        leg_h.EdgePixelObject,
        leg_h.LSTObject,
        leg_h.MSTObject,
        leg_h.MeanRadiusOuterEdgeObject,
        leg_h.OffPixelObject,
        leg_h.PixelObject,
        leg_h.SCTObject,
        leg_h.SSTObject,
    ]

    for obj in object_list:
        instance = obj()
        assert isinstance(instance, object)


def test_handlers(io_handler):
    handler_list = [
        leg_h.HexEdgePixelHandler,
        leg_h.HexOffPixelHandler,
        leg_h.HexPixelHandler,
        leg_h.LSTHandler,
        leg_h.MSTHandler,
        leg_h.MeanRadiusOuterEdgeHandler,
        leg_h.SCTHandler,
        leg_h.SquareEdgePixelHandler,
        leg_h.SquareOffPixelHandler,
        leg_h.SquarePixelHandler,
    ]

    for handler in handler_list:
        handler_instance = handler()
        assert handler_instance.legend_artist is not None

    tel_handler = leg_h.TelescopeHandler("LSTN")
    assert tel_handler.config is not None
    assert "color" in tel_handler.config
    assert "radius" in tel_handler.config
    assert "shape" in tel_handler.config


def test_telescope_config():
    """Test that all telescope types in TELESCOPE_CONFIG work with TelescopeHandler."""
    for telescope_type in leg_h.TELESCOPE_CONFIG:
        handler = leg_h.TelescopeHandler(telescope_type)
        assert handler.telescope_type == telescope_type
        assert handler.config == leg_h.TELESCOPE_CONFIG[telescope_type]


def test_get_telescope_config():
    """Test that get_telescope_config returns the correct configuration."""
    for telescope_type in leg_h.TELESCOPE_CONFIG:
        config = leg_h.get_telescope_config(telescope_type)
        assert config == leg_h.TELESCOPE_CONFIG[telescope_type]

    prefix = "LST"
    config = leg_h.get_telescope_config(prefix)
    assert config == leg_h.TELESCOPE_CONFIG["LST"]


def test_calculate_center():
    handlebox = MockHandleBox(xdescent=10, ydescent=20, width=30, height=40)

    x0, y0 = leg_h.calculate_center(handlebox, width_factor=3, height_factor=4)
    assert x0 == 10 + 30 / 3
    assert y0 == 20 + 40 / 4

    x0, y0 = leg_h.calculate_center(handlebox, width_factor=2, height_factor=2)
    assert x0 == 10 + 30 / 2
    assert y0 == 20 + 40 / 2


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
    """Test the TelescopeHandler class for various telescope types."""
    for telescope_type, config in leg_h.TELESCOPE_CONFIG.items():
        handler = leg_h.TelescopeHandler(telescope_type)
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


def test_mean_radius_outer_edge_handler():
    """Test the MeanRadiusOuterEdgeHandler class."""
    handlebox = MockHandleBox()
    handler = leg_h.MeanRadiusOuterEdgeHandler()
    patch = handler.legend_artist(None, None, None, handlebox)

    assert patch is not None
    assert isinstance(patch, leg_h.mpatches.Circle)
    assert patch.get_facecolor() == (0.0, 0.0, 0.0, 0.0)
    assert patch.get_edgecolor() == (*leg_h.mcolors.to_rgb("darkorange"), 1)
    assert patch.get_radius() == handlebox.height
    assert handlebox.artist == patch
