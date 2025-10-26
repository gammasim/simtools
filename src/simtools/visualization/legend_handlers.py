"""Helper functions for legend handlers used for plotting."""

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

"""
Define properties of different telescope types for visualization purposes.

Radii are relative to a reference radius (REFERENCE_RADIUS).
"""
TELESCOPE_CONFIG = {
    "LST": {"color": "darkorange", "radius": 12.5, "shape": "circle", "filled": False},
    "MST": {"color": "dodgerblue", "radius": 9.15, "shape": "circle", "filled": False},
    "SCT": {"color": "black", "radius": 7.15, "shape": "square", "filled": False},
    "SST": {"color": "darkgreen", "radius": 3.0, "shape": "circle", "filled": False},
    "HESS": {"color": "grey", "radius": 6.0, "shape": "hexagon", "filled": True},
    "MAGIC": {"color": "grey", "radius": 8.5, "shape": "hexagon", "filled": True},
    "VERITAS": {"color": "grey", "radius": 6.0, "shape": "hexagon", "filled": True},
    "CEI": {"color": "purple", "radius": 2.0, "shape": "hexagon", "filled": True},
    "RLD": {"color": "brown", "radius": 2.0, "shape": "hexagon", "filled": True},
    "STP": {"color": "olive", "radius": 2.0, "shape": "hexagon", "filled": True},
    "MSP": {"color": "teal", "radius": 2.0, "shape": "hexagon", "filled": True},
    "ILL": {"color": "red", "radius": 2.0, "shape": "hexagon", "filled": False},
    "WST": {"color": "maroon", "radius": 2.0, "shape": "hexagon", "filled": True},
    "ASC": {"color": "cyan", "radius": 2.0, "shape": "hexagon", "filled": True},
    "DUS": {"color": "magenta", "radius": 2.0, "shape": "hexagon", "filled": True},
}

REFERENCE_RADIUS = 12.5


def get_telescope_config(telescope_type):
    """
    Return the configuration for a given telescope type.

    Try both site-dependent and site-independent configurations (e.g. "MSTS" and "MST").

    Parameters
    ----------
    telescope_type : str, None
        The type of the telescope (e.g., "LSTN", "MSTS").

    Returns
    -------
    dict
        The configuration dictionary for the telescope type.
    """
    if telescope_type is None:
        return {"color": "blue", "radius": 2.0, "shape": "hexagon", "filled": True}
    config = TELESCOPE_CONFIG.get(telescope_type)
    if not config and len(telescope_type) >= 3:
        config = TELESCOPE_CONFIG.get(telescope_type[:3])
    return config


def calculate_center(handlebox, width_factor=3, height_factor=3):
    """Calculate the center of the handlebox based on given factors."""
    x0 = handlebox.xdescent + handlebox.width / width_factor
    y0 = handlebox.ydescent + handlebox.height / height_factor
    return x0, y0


# Object classes for legend mapping
class PixelObject:
    """Pixel Object."""


class EdgePixelObject:
    """Edge-Pixel Object."""


class OffPixelObject:
    """Off-Pixel Object."""


class LSTObject:
    """LST Object."""


class MSTObject:
    """MST Object."""


class SCTObject:
    """SCT Object."""


class SSTObject:
    """SST Object."""


class HESSObject:
    """HESS Object."""


class MAGICObject:
    """MAGIC Object."""


class VERITASObject:
    """VERITAS Object."""


class MeanRadiusOuterEdgeObject:
    """Object for Mean radius outer edge."""


# Pixel handlers
class _BaseHexPixelHandler:
    """Base class for hexagonal pixel handlers."""

    @staticmethod
    def _create_hex_patch(handlebox, facecolor, edgecolor):
        """Create a hexagonal patch with specified colors."""
        x0, y0 = calculate_center(handlebox)
        patch = mpatches.RegularPolygon(
            (x0, y0),
            numVertices=6,
            radius=0.7 * handlebox.height,
            orientation=np.deg2rad(30),
            facecolor=facecolor,
            edgecolor=edgecolor,
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class HexPixelHandler(_BaseHexPixelHandler):
    """Legend handler class to plot a hexagonal "on" pixel."""

    @staticmethod
    def legend_artist(_, __, ___, handlebox):
        """Legend artist."""
        return HexPixelHandler._create_hex_patch(
            handlebox, facecolor=(1, 1, 1, 0), edgecolor=(0, 0, 0, 1)
        )


class HexEdgePixelHandler(_BaseHexPixelHandler):
    """Legend handler class to plot a hexagonal "edge" pixel."""

    @staticmethod
    def legend_artist(_, __, ___, handlebox):
        """Legend artist."""
        return HexEdgePixelHandler._create_hex_patch(
            handlebox,
            facecolor=(*mcolors.to_rgb("brown"), 0.5),
            edgecolor=(*mcolors.to_rgb("black"), 1),
        )


class HexOffPixelHandler(_BaseHexPixelHandler):
    """Legend handler class to plot a hexagonal "off" pixel."""

    @staticmethod
    def legend_artist(_, __, ___, handlebox):
        """Legend artist."""
        return HexOffPixelHandler._create_hex_patch(handlebox, facecolor="black", edgecolor="black")


class _BaseSquarePixelHandler:
    """Base class for square pixel handlers."""

    @staticmethod
    def _create_square_patch(handlebox, facecolor, edgecolor):
        """Create a square patch with specified colors."""
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width = height = handlebox.height
        patch = mpatches.Rectangle(
            [x0, y0],
            width,
            height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class SquarePixelHandler(_BaseSquarePixelHandler):
    """Legend handler class to plot a square "on" pixel."""

    @staticmethod
    def legend_artist(_, __, ___, handlebox):
        """Legend artist."""
        return SquarePixelHandler._create_square_patch(
            handlebox, facecolor=(1, 1, 1, 0), edgecolor=(0, 0, 0, 1)
        )


class SquareEdgePixelHandler(_BaseSquarePixelHandler):
    """Legend handler class to plot a square "edge" pixel."""

    @staticmethod
    def legend_artist(_, __, ___, handlebox):
        """Legend artist."""
        return SquareEdgePixelHandler._create_square_patch(
            handlebox,
            facecolor=(*mcolors.to_rgb("brown"), 0.5),
            edgecolor=(*mcolors.to_rgb("black"), 1),
        )


class SquareOffPixelHandler(_BaseSquarePixelHandler):
    """Legend handler class to plot a square "off" pixel."""

    @staticmethod
    def legend_artist(_, __, ___, handlebox):
        """Legend artist."""
        return SquareOffPixelHandler._create_square_patch(
            handlebox, facecolor="black", edgecolor="black"
        )


class BaseLegendHandler:
    """Base telescope handler that can handle any telescope type."""

    def __init__(self, telescope_type):
        self.telescope_type = telescope_type
        self.config = get_telescope_config(telescope_type)

    def _create_circle(self, handlebox, x0, y0, radius):
        """Create a circle patch."""
        facecolor = self.config["color"] if self.config["filled"] else "none"
        return mpatches.Circle(
            xy=(x0, y0),
            radius=radius * self.config["radius"] / REFERENCE_RADIUS,
            facecolor=facecolor,
            edgecolor=self.config["color"],
            transform=handlebox.get_transform(),
        )

    def _create_square(self, handlebox, x0, y0, size):
        """Create a square patch."""
        return mpatches.Rectangle(
            [x0, y0],
            size,
            size,
            facecolor=self.config["color"],
            edgecolor=self.config["color"],
            transform=handlebox.get_transform(),
        )

    def _create_hexagon(self, handlebox, x0, y0, radius):
        """Create a hexagon patch."""
        return mpatches.RegularPolygon(
            (x0, y0),
            numVertices=6,
            radius=0.7 * radius * self.config["radius"] / REFERENCE_RADIUS,
            orientation=np.deg2rad(30),
            facecolor=self.config["color"],
            edgecolor=self.config["color"],
            transform=handlebox.get_transform(),
        )

    def legend_artist(self, _, __, ___, handlebox):
        """Create the appropriate patch based on telescope type."""
        shape = self.config["shape"]

        if shape == "circle":
            x0, y0 = calculate_center(handlebox, 4, 2)
            radius = handlebox.height
            patch = self._create_circle(handlebox, x0, y0, radius)
        elif shape == "square":
            x0, y0 = calculate_center(handlebox, 10, 1)
            size = handlebox.height
            patch = self._create_square(handlebox, x0, y0, size)
        elif shape == "hexagon":
            x0, y0 = calculate_center(handlebox)
            radius = handlebox.height
            patch = self._create_hexagon(handlebox, x0, y0, radius)

        handlebox.add_artist(patch)
        return patch


class LSTHandler(BaseLegendHandler):
    """Legend handler for LST telescopes."""

    def __init__(self):
        super().__init__("LST")


class MSTHandler(BaseLegendHandler):
    """Legend handler for MST telescopes."""

    def __init__(self):
        super().__init__("MST")


class SSTHandler(BaseLegendHandler):
    """Legend handler for SST telescopes."""

    def __init__(self):
        super().__init__("SST")


class SCTHandler(BaseLegendHandler):
    """Legend handler for SCT telescopes."""

    def __init__(self):
        super().__init__("SCT")


class HESSHandler(BaseLegendHandler):
    """Legend handler for HESS telescopes."""

    def __init__(self):
        super().__init__("HESS")


class MAGICHandler(BaseLegendHandler):
    """Legend handler for MAGIC telescopes."""

    def __init__(self):
        super().__init__("MAGIC")


class VERITASHandler(BaseLegendHandler):
    """Legend handler for VERITAS telescopes."""

    def __init__(self):
        super().__init__("VERITAS")


class MeanRadiusOuterEdgeHandler:
    """Legend handler class to plot the mean radius outer edge of the dish."""

    @staticmethod
    def legend_artist(_, __, ___, handlebox):  # noqa: D102
        x0, y0 = calculate_center(handlebox, 4, 4)
        radius = handlebox.height
        patch = mpatches.Circle(
            xy=(x0, y0),
            radius=radius,
            facecolor="none",
            edgecolor="darkorange",
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


legend_handler_map = {
    LSTObject: LSTHandler,
    MSTObject: MSTHandler,
    SSTObject: SSTHandler,
    SCTObject: SCTHandler,
    HESSObject: HESSHandler,
    MAGICObject: MAGICHandler,
    VERITASObject: VERITASHandler,
}
