import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np


class pixelObject(object):
    pass


class edgePixelObject(object):
    pass


class offPixelObject(object):
    pass


class lstObject(object):
    pass


class mstObject(object):
    pass


class sstObject(object):
    pass


class meanRadiusOuterEdgeObject(object):
    pass


class hexPixelHandler(object):
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


class hexEdgePixelHandler(object):
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


class hexOffPixelHandler(object):
    """
    Legend handler class to plot a hexagonal "off" pixel.
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
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


class squarePixelHandler(object):
    """
    Legend handler class to plot a square "on" pixel.
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
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


class squareEdgePixelHandler(object):
    """
    Legend handler class to plot a square "edge" pixel.
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
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


class squareOffPixelHandler(object):
    """
    Legend handler class to plot a square "off" pixel.
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
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


class lstHandler(object):
    """
    Legend handler class to plot a representation of an LST in an array layout.
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
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


class mstHandler(object):
    """
    Legend handler class to plot a representation of an MST in an array layout.
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
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


class sstHandler(object):
    """
    Legend handler class to plot a representation of an SST in an array layout.
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
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


class meanRadiusOuterEdgeHandler(object):
    """
    Legend handler class to plot a the mean radius outer edge of the dish.
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
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
