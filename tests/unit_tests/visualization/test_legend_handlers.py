import pytest

from simtools.utils import names
from simtools.visualization import legend_handlers as leg_h


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
    colors = ["darkorange", "dodgerblue", "black", "darkgreen", "grey", "grey", "grey"]
    radius_dict = [12.5, 9.15, 7.15, 3, 7.5, 10, 9.15]
    for step, tel_type in enumerate(names.all_telescope_class_names):
        assert tel_handler.radius_dict[tel_type] == pytest.approx(radius_dict[step], 1.0e-3)
        assert tel_handler.colors_dict[tel_type] == colors[step]
