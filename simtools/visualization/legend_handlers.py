import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

from simtools.corsika.corsika_config import CorsikaConfig
from simtools.io_handler import IOHandler
from simtools.util import names
from simtools.util.names import lst, mst, sst

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
    "SCTHandler",
    "SCTObject",
    "SSTHandler",
    "SSTObject",
    "SquareEdgePixelHandler",
    "SquareOffPixelHandler",
    "SquarePixelHandler",
    "TelescopeHandler",
]


class TelescopeHandler(object):
    """
    Telescope handler that centralizes the telescope information. Individual telescopes handlers
    inherit the objects from this class.
    """

    def __init__(self):
        io_handler = IOHandler()
        corsika_parameters_file = io_handler.get_input_data_file(
            "parameters", "corsika_parameters.yml"
        )
        corsika_info = CorsikaConfig.load_corsika_parameters_file(corsika_parameters_file)

        self.radius_dict = {}
        for _, tel_type in enumerate(names.all_telescope_class_names):
            self.radius_dict[tel_type] = corsika_info["corsika_sphere_radius"][tel_type]["value"]


class PixelObject(object):
    """Pixel Object."""


class EdgePixelObject(object):
    """Edge-Pixel Object."""


class OffPixelObject(object):
    """Off-Pixel Object."""


class LSTObject(object):
    """LST Object."""


class MSTObject(object):
    """MST Object."""


class SCTObject(object):
    """SCT Object."""


class SSTObject(object):
    """SST Object."""


class MeanRadiusOuterEdgeObject(object):
    """Object for Mean radius outer edge."""


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


class LSTHandler(TelescopeHandler):
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
            radius=radius * self.radius_dict[lst] / self.radius_dict[lst],
            facecolor="none",
            edgecolor="darkorange",
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class MSTHandler(TelescopeHandler):
    """
    Legend handler class to plot a representation of an MST in an array layout.
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        center = (
            handlebox.xdescent + 0.25 * handlebox.width,
            handlebox.ydescent + 0.5 * handlebox.height,
        )
        radius = handlebox.height
        patch = mpatches.Circle(
            xy=center,
            radius=radius * self.radius_dict[mst] / self.radius_dict[lst],
            facecolor="dodgerblue",
            edgecolor="dodgerblue",
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class SSTHandler(TelescopeHandler):
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
            radius=radius * self.radius_dict[sst] / self.radius_dict[lst],
            facecolor="black",
            edgecolor="black",
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class SCTHandler(object):
    """
    Legend handler class to plot a representation of an SCT in an array layout.
    """

    @staticmethod
    def legend_artist(legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent + 0.1 * handlebox.width, handlebox.ydescent
        width = height = handlebox.height
        patch = mpatches.Rectangle(
            [x0, y0],
            width,
            height,
            facecolor="lightsteelblue",
            edgecolor="lightsteelblue",
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


all_telescope_objects = [LSTObject, MSTObject, SCTObject, SSTObject]
all_telescope_handlers = [LSTHandler, MSTHandler, SCTHandler, SSTHandler]
legend_handler_map = {
    telescope_object: telescope_handler
    for telescope_object, telescope_handler in zip(all_telescope_objects, all_telescope_handlers)
}
