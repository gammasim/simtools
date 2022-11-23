import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

__all__ = [
    "EdgePixelObject",
    "HexEdgePixelHandler",
    "HexOffPixelHandler",
    "HexPixelHandler",
    "LSTHandler",
    "LSTObject",
    "MSTHandler",
    "MSTObject",
    "MeanRadiusOuterEdgeHandler",
    "MeanRadiusOuterEdgeObject",
    "OffPixelObject",
    "PixelObject",
    "SSTHandler",
    "SSTObject",
    "SquareEdgePixelHandler",
    "SquareOffPixelHandler",
    "SquarePixelHandler",
]


class PixelObject(object):
    """Pixel Object."""

    pass


class EdgePixelObject(object):
    """Edge-Pixel Object."""

    pass


class OffPixelObject(object):
    """Off-Pixel Object."""

    pass


class LSTObject(object):
    """LST Object."""

    pass


class MSTObject(object):
    """MST Object."""

    pass


class SSTObject(object):
    """SST Object."""

    pass


class MeanRadiusOuterEdgeObject(object):
    """Object for Mean radius outer edge."""

    pass


class HexPixelHandler(object):
    """
    Legend handler class to plot a hexagonal "on" pixel.
    """

    @staticmethod
    def legend_artist(legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent + handlebox.width / 3, handlebox.ydescent + handlebox.height / 3
        # width = height = handlebox.height
        patch = mpatches.RegularPolygon(
            (x0, y0),
            numVertices=6,
            radius=0.7 * handlebox.height,
            orientation=np.deg2rad(30),
            facecolor=(1, 1, 1, 0),
            edgecolor=(0, 0, 0, 1),
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class HexEdgePixelHandler(object):
    """
    Legend handler class to plot a hexagonal "edge" pixel.
    """

    @staticmethod
    def legend_artist(legend, orig_handle, fontsize, handlebox):
        x0, y0 = (
            handlebox.xdescent + handlebox.width / 3,
            handlebox.ydescent + handlebox.height / 3,
        )
        # width = height = handlebox.height
        patch = mpatches.RegularPolygon(
            (x0, y0),
            numVertices=6,
            radius=0.7 * handlebox.height,
            orientation=np.deg2rad(30),
            facecolor=mcolors.to_rgb("brown") + (0.5,),
            edgecolor=mcolors.to_rgb("black") + (1,),
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class HexOffPixelHandler(object):
    """
    Legend handler class to plot a hexagonal "off" pixel.
    """

    @staticmethod
    def legend_artist(legend, orig_handle, fontsize, handlebox):
        x0, y0 = (
            handlebox.xdescent + handlebox.width / 3,
            handlebox.ydescent + handlebox.height / 3,
        )
        # width = height = handlebox.height
        patch = mpatches.RegularPolygon(
            (x0, y0),
            numVertices=6,
            radius=0.7 * handlebox.height,
            orientation=np.deg2rad(30),
            facecolor="black",
            edgecolor="black",
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class SquarePixelHandler(object):
    """
    Legend handler class to plot a square "on" pixel.
    """

    @staticmethod
    def legend_artist(legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width = height = handlebox.height
        patch = mpatches.Rectangle(
            [x0, y0],
            width,
            height,
            facecolor=(1, 1, 1, 0),
            edgecolor=(0, 0, 0, 1),
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class SquareEdgePixelHandler(object):
    """
    Legend handler class to plot a square "edge" pixel.
    """

    @staticmethod
    def legend_artist(legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width = height = handlebox.height
        patch = mpatches.Rectangle(
            [x0, y0],
            width,
            height,
            facecolor=mcolors.to_rgb("brown") + (0.5,),
            edgecolor=mcolors.to_rgb("black") + (1,),
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class SquareOffPixelHandler(object):
    """
    Legend handler class to plot a square "off" pixel.
    """

    @staticmethod
    def legend_artist(legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width = height = handlebox.height
        patch = mpatches.Rectangle(
            [x0, y0],
            width,
            height,
            facecolor="black",
            edgecolor="black",
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class LSTHandler(object):
    """
    Legend handler class to plot a representation of an LST in an array layout.
    """

    @staticmethod
    def legend_artist(legend, orig_handle, fontsize, handlebox):
        center = (
            handlebox.xdescent + 0.3 * handlebox.width,
            handlebox.ydescent + 0.5 * handlebox.height,
        )
        radius = handlebox.height
        patch = mpatches.Circle(
            xy=center,
            radius=radius,
            facecolor="none",
            edgecolor="darkorange",
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class MSTHandler(object):
    """
    Legend handler class to plot a representation of an MST in an array layout.
    """

    @staticmethod
    def legend_artist(legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent + 0.1 * handlebox.width, handlebox.ydescent
        width = height = handlebox.height
        patch = mpatches.Rectangle(
            [x0, y0],
            width,
            height,
            facecolor="dodgerblue",
            edgecolor="dodgerblue",
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class SSTHandler(object):
    """
    Legend handler class to plot a representation of an SST in an array layout.
    """

    @staticmethod
    def legend_artist(legend, orig_handle, fontsize, handlebox):
        center = (
            handlebox.xdescent + 0.25 * handlebox.width,
            handlebox.ydescent + 0.5 * handlebox.height,
        )
        radius = handlebox.height
        patch = mpatches.Circle(
            xy=center,
            radius=radius * (2.8 / 12),
            facecolor="black",
            edgecolor="black",
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class MeanRadiusOuterEdgeHandler(object):
    """
    Legend handler class to plot a the mean radius outer edge of the dish.
    """

    @staticmethod
    def legend_artist(legend, orig_handle, fontsize, handlebox):
        center = (
            handlebox.xdescent + 0.25 * handlebox.width,
            handlebox.ydescent + 0.25 * handlebox.height,
        )
        radius = handlebox.height
        patch = mpatches.Circle(
            xy=center,
            radius=radius,
            facecolor="none",
            edgecolor="darkorange",
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch
