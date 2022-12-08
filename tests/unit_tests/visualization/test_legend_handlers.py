from simtools.visualization import legend_handlers as leg_h


def test_objects():
    object_list = [
        leg_h.EdgePixelObject,
        leg_h.LSTObject,
        leg_h.MSTObject,
        leg_h.MeanRadiusOuterEdgeObject,
        leg_h.OffPixelObject,
        leg_h.PixelObject,
        leg_h.SSTObject,
    ]

    for obj in object_list:
        instance = obj()
        assert isinstance(instance, object)


def test_handlers():
    handler_list = [
        leg_h.HexEdgePixelHandler,
        leg_h.HexOffPixelHandler,
        leg_h.HexPixelHandler,
        leg_h.LSTHandler,
        leg_h.MSTHandler,
        leg_h.MeanRadiusOuterEdgeHandler,
        leg_h.SquareEdgePixelHandler,
        leg_h.SquareOffPixelHandler,
        leg_h.SquarePixelHandler,
    ]
    for handler in handler_list:
        instance = handler()
        assert isinstance(instance, object)
        assert instance.legend_artist is not None
