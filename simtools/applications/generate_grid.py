"""
Module defines the `GridGeneration` class.

Used to generate a grid of simulation points based on flexible axes definitions.
The grid can be adapted based on different science cases and data levels. The
axes include parameters energy, azimuth, zenith angle, night-sky background,
and camera offset. The class handles various axis scaling, binning,
and distribution types, allowing for adaptable simulation configurations.

Key Components:
---------------
- `GridGeneration`: Class to handle the generation and adaptation of grid points for simulations.
  - Attributes:
    - `axes` (list of dict): List of dictionaries defining each axis with
    properties such as name, range, binning, scaling, etc.
    - `data_level` (str): The data level for the grid generation (e.g., 'A', 'B', 'C').
    - `science_case` (str): The science case for the grid generation (e.g., 'high_precision').

Example Usage:
--------------
```python
axes = [
    {"name": "energy", "range": (1e9, 1e14), "binning": 10, "scaling": "log",
      "distribution": "uniform"},
    {"name": "azimuth", "range": (70, 80), "binning": 3, "scaling": "linear",
      "distribution": "uniform"},
    {"name": "zenith_angle", "range": (20, 30), "binning": 3, "scaling": "linear",
      "distribution": "uniform"},
    {"name": "night_sky_background", "range": (5, 6), "binning": 2, "scaling": "linear",
      "distribution": "uniform"},
    {"name": "offset", "range": (0, 10), "binning": 5, "scaling": "linear",
      "distribution": "uniform"},
]

# Define data level and science case
data_level = "B"
science_case = "high_precision"

# Create grid generation instance
grid_gen = GridGeneration(axes, data_level, science_case)

# Generate grid points
grid_points = grid_gen.generate_grid()

"""

import numpy as np


class GridGeneration:
    """
    Defines and generates a grid of simulation points based on flexible axes definitions.

    This class generates a grid of points for a simulation based on parameters like energy, azimuth,
    zenith angle, night-sky background, and camera offset. The grid adapts to
    different science cases and data levels,
      taking into account flexible axis definitions and scaling.

    Key Components:
    ---------------
    - `GridGeneration`: Main class to handle the generation and adaptation of grid
      points for simulations.
      - Attributes:
        - `axes` (list of dict): List of dictionaries defining each axis with
          properties such as name, range, binning, scaling, etc.
        - `data_level` (str): The data level for the grid generation (e.g., 'A', 'B', 'C').
        - `science_case` (str): The science case for the grid generation (e.g., 'high_precision').
    """

    def __init__(self, axes: list[dict], data_level: str, science_case: str):
        """
        Initialize the grid generation with the given axes, data level, and science case.

        Parameters
        ----------
        axes : list of dict
            List of dictionaries where each dictionary defines an axis with properties
              such as name, range, binning, scaling, and distribution type.
        data_level : str
            The data level (e.g., 'A', 'B', 'C') for the grid generation.
        science_case : str
            The science case for the grid generation (e.g., 'high_precision').
        """
        self.axes = axes
        self.data_level = data_level
        self.science_case = science_case

    def generate_grid(self) -> list[dict]:
        """
        Generate the grid based on the defined axes, data level, and science case.

        Returns
        -------
        list of dict
            A list of grid points, each represented as a dictionary with axis names as keys and
              axis values as values.
        """
        axis_values = {}

        for axis in self.axes:
            name = axis["name"]
            axis_range = axis["range"]
            binning = axis["binning"]
            scaling = axis.get("scaling", "linear")
            # distribution = axis.get('distribution', 'uniform') # Add

            # Create axis values based on scaling and distribution
            if scaling == "log":
                values = np.logspace(np.log10(axis_range[0]), np.log10(axis_range[1]), binning)
            else:
                values = np.linspace(axis_range[0], axis_range[1], binning)

            axis_values[name] = values

        # Generate all combinations of axis values
        return [
            dict(zip(axis_values.keys(), values))
            for values in np.array(np.meshgrid(*axis_values.values())).T.reshape(
                -1, len(axis_values)
            )
        ]

    def adapt_grid(self) -> list[dict]:
        """
        Adapt the grid definition based on the science case and data level.

        Returns
        -------
        list of dict
            The adapted list of grid points based on the science case and data level.
        """
        adapted_axes = []

        for axis in self.axes:
            adapted_axis = axis.copy()

            if self.science_case == "high_precision":
                adapted_axis["binning"] *= 2  # Increase binning for high precision
            elif self.science_case == "low_precision":
                adapted_axis["binning"] //= 2  # Decrease binning for low precision

            # Adapt axis range based on data level
            if self.data_level == "A":
                adapted_axis["range"] = (
                    axis["range"][0],
                    axis["range"][1] * 0.5,
                )  # Example adjustment
            elif self.data_level == "B":
                adapted_axis["range"] = (axis["range"][0], axis["range"][1])
            elif self.data_level == "C":
                adapted_axis["range"] = (
                    axis["range"][0],
                    axis["range"][1] * 1.5,
                )  # Example adjustment

            adapted_axes.append(adapted_axis)

        # Regenerate the grid with adapted parameters
        self.axes = adapted_axes
        return self.generate_grid()
