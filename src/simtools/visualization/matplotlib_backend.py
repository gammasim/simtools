"""Configure the Matplotlib backend used by simtools visualizations."""

from importlib import import_module

import matplotlib as mpl

# The backend must be selected before pyplot is imported. Using import_module
# here keeps that ordering explicit without placing an import after executable
# code in every plotting module.
mpl.use("Agg")
pyplot = import_module("matplotlib.pyplot")
