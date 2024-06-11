#!/usr/bin/python3

from simtools.utils import names
from simtools.visualization import legend_handlers as leg_h
from simtools.visualization.legend_handlers import SquareOffPixelHandler


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

    tel_handler = leg_h.TelescopeHandler()
    assert len(tel_handler.colors_dict) == len(
        names.get_list_of_telescope_types(
            array_element_class="telescopes", site=None, observatory=None
        )
    )


def test_legend_artist_handlebox_none():
    # Call the legend_artist method with handlebox=None
    result = SquareOffPixelHandler.legend_artist(handlebox=None)

    # Assert that the result is None
    assert result is None
