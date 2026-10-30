"""Helper functions for legend handlers used for plotting."""

# pylint: disable=too-few-public-methods

import numpy as np

from simtools.visualization.matplotlib_backend import lazy_module

mcolors = lazy_module("matplotlib.colors")
mpatches = lazy_module("matplotlib.patches")

# Properties of telescope types for visualization. Radii are relative to REFERENCE_RADIUS.
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
    return config.copy() if config else None


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


class _BaseHexPixelHandler:
    """Base class for hexagonal pixel handlers; subclasses set ``colors`` (face, edge)."""

    colors = ("black", "black")

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

    def legend_artist(self, _, __, ___, handlebox):
        """Legend artist."""
        return self._create_hex_patch(handlebox, *self.colors)


class HexPixelHandler(_BaseHexPixelHandler):
    """Legend handler class to plot a hexagonal "on" pixel."""

    colors = ((1, 1, 1, 0), (0, 0, 0, 1))


class HexEdgePixelHandler(_BaseHexPixelHandler):
    """Legend handler class to plot a hexagonal "edge" pixel."""

    colors = ((0.6470588235294118, 0.16470588235294117, 0.16470588235294117, 0.5), (0, 0, 0, 1))


class HexOffPixelHandler(_BaseHexPixelHandler):
    """Legend handler class to plot a hexagonal "off" pixel."""

    colors = ("black", "black")


class _BaseSquarePixelHandler:
    """Base class for square pixel handlers; subclasses set ``colors`` (face, edge)."""

    colors = ("black", "black")

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

    def legend_artist(self, _, __, ___, handlebox):
        """Legend artist."""
        return self._create_square_patch(handlebox, *self.colors)


class SquarePixelHandler(_BaseSquarePixelHandler):
    """Legend handler class to plot a square "on" pixel."""

    colors = ((1, 1, 1, 0), (0, 0, 0, 1))


class SquareEdgePixelHandler(_BaseSquarePixelHandler):
    """Legend handler class to plot a square "edge" pixel."""

    colors = ((0.6470588235294118, 0.16470588235294117, 0.16470588235294117, 0.5), (0, 0, 0, 1))


class SquareOffPixelHandler(_BaseSquarePixelHandler):
    """Legend handler class to plot a square "off" pixel."""

    colors = ("black", "black")


class BaseLegendHandler:
    """Base telescope handler that can handle any telescope type."""

    telescope_type = None

    def __init__(self, telescope_type=None):
        self.telescope_type = self.telescope_type if telescope_type is None else telescope_type
        self.config = get_telescope_config(self.telescope_type)

    def _create_patch(self, handlebox, x0, y0, size):
        """Create the patch for the configured shape, sized relative to ``size``."""
        color = self.config["color"]
        transform = handlebox.get_transform()
        shape = self.config["shape"]
        if shape == "circle":
            facecolor = color if self.config["filled"] else "none"
            return mpatches.Circle(
                xy=(x0, y0),
                radius=size * self.config["radius"] / REFERENCE_RADIUS,
                facecolor=facecolor,
                edgecolor=color,
                transform=transform,
            )
        if shape == "square":
            return mpatches.Rectangle(
                [x0, y0], size, size, facecolor=color, edgecolor=color, transform=transform
            )
        if shape == "hexagon":
            return mpatches.RegularPolygon(
                (x0, y0),
                numVertices=6,
                radius=0.7 * size * self.config["radius"] / REFERENCE_RADIUS,
                orientation=np.deg2rad(30),
                facecolor=color,
                edgecolor=color,
                transform=transform,
            )
        raise ValueError(f"Unknown shape: {shape}")

    def legend_artist(self, _, __, ___, handlebox):
        """Create the appropriate patch based on telescope type."""
        shape = self.config["shape"]
        if shape == "circle":
            x0, y0 = calculate_center(handlebox, 4, 2)
        elif shape == "square":
            x0, y0 = calculate_center(handlebox, 10, 1)
        else:
            x0, y0 = calculate_center(handlebox)
        patch = self._create_patch(handlebox, x0, y0, handlebox.height)
        handlebox.add_artist(patch)
        return patch


class LSTHandler(BaseLegendHandler):
    """Legend handler for LST telescopes."""

    telescope_type = "LST"


class MSTHandler(BaseLegendHandler):
    """Legend handler for MST telescopes."""

    telescope_type = "MST"


class SSTHandler(BaseLegendHandler):
    """Legend handler for SST telescopes."""

    telescope_type = "SST"


class SCTHandler(BaseLegendHandler):
    """Legend handler for SCT telescopes."""

    telescope_type = "SCT"


class HESSHandler(BaseLegendHandler):
    """Legend handler for HESS telescopes."""

    telescope_type = "HESS"


class MAGICHandler(BaseLegendHandler):
    """Legend handler for MAGIC telescopes."""

    telescope_type = "MAGIC"


class VERITASHandler(BaseLegendHandler):
    """Legend handler for VERITAS telescopes."""

    telescope_type = "VERITAS"
