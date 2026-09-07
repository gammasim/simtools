# simtools-plot-array-layout

```{eval-rst}
.. automodule:: simtools.applications.plot_array_layout
   :members:
   :exclude-members: main
```

```{eval-rst}
Plot array layouts in ground or UTM coordinate systems from multiple sources.

For the following options, array element positions are retrieved from the model parameter database:

* from the model parameter database using the layout name (e.g., ``-array_layout_name alpha``)

* from the model parameter data, retrieving all layouts for the given site and model version
  (``--plot_all_layouts``)

* from a model parameter file
  (e.g., ``-array_layout_parameter_file tests/resources/model_parameters/array_layouts-2.0.2.json``)

* from a list of array elements (e.g., ``-array_element_list MSTN-01, MSTN-02``).
  Positions are retrieved from the database.
  * explicit listing: e.g., ``-array_element_list MSTN-01, MSTN05``
  * listing of types: e.g, ``-array_element_list MSTN`` plots all telescopes of type MSTN.

For this option, array element positions are retrieved from the input file:

* from a file containing an astropy table with a list of array elements and their positions
  (e.g., ``-array_layout_file tests/resources/telescope_positions-North-ground.ecsv``)

Plots are saved as png files in the output directory by default.

Example of a layout plot:

.. _plot_array_layout_plot:
.. image:: images/plot_array_layout_example.png
    :width: 49 %

**Command line arguments**

figure_name : str
    File name for the output figure.
array_layout_file : str
    File (astropy table compatible) with a list of array elements.
array_layout_name : str
    Name of the layout array (e.g., test_layout, alpha, 4mst, etc.).
    Use 'plot_all' to plot all layouts from the database for the given site and model version.
array_layout_parameter_file : str, optional
    File with array layouts similar in the model parameter file format (typically JSON).
array_layout_name_from_parameter_file : list, optional
    Name(s) of the array layout(s) to plot from ``array_layout_parameter_file``.
array_layout_name_background: str, optional
    Name of the background layout array (e.g., test_layout, alpha, 4mst, etc.).
array_element_list : list
    List of array elements (e.g., telescopes) to plot (e.g., ``LSTN-01 LSTN-02 MSTN``).
coordinate_system : str, optional
    Coordinate system for the array layout (ground or utm).
show_labels : bool, optional
    Shows the telescope labels in the plot.
axes_range : float, optional
    Range of the both axes in meters.
marker_scaling : float, optional.
    Scaling factor for plotting of array elements, optional.
grayed_out_array_elements : list, optional
    List of array elements to plot as gray circles.
highlighted_array_elements : list, optional
    List of array elements to plot with red circles around them.
legend_location : str, optional
    Location of the legend (default "best").
bounds : str, optional
    Axis bounds mode. Use "symmetric" for +-R with padding (default) or "exact" for
    per-axis min/max bounds.
padding : float, optional
    Fractional padding applied around computed extents in both modes (default 0.1).
x_lim : tuple(float, float), optional
    Explicit x-axis limits [xmin, xmax] in meters. When provided, overrides derived limits
    and filters plotted elements by x.
y_lim : tuple(float, float), optional
    Explicit y-axis limits [ymin, ymax] in meters. When provided, overrides derived limits
    and filters plotted elements by y.

**Examples**

Plot "alpha" layout for the North site with model version 6.0.0:

.. code-block:: console

    simtools-plot-array-layout --site North
                               --array_layout_name alpha
                               --model_version=6.0.0

Plot layout with 2 LSTs on top of north alpha layout:

.. code-block:: console

    simtools-plot-array-layout --site North
                               --array_element_list LSTN-01,LSTN-02
                               --model_version=6.0.0
                               --array_layout_name_background alpha

Plot layout from a file with a list of telescopes:

.. code-block:: console

    simtools-plot-array-layout
        --array_layout_file tests/resources/telescope_positions-North-ground.ecsv

Use exact bounds with default padding:

.. code-block:: console

    simtools-plot-array-layout --array_layout_name alpha         --site North --model_version 6.0.0 --bounds exact

Use symmetric bounds with custom padding:

.. code-block:: console

    simtools-plot-array-layout --array_layout_name alpha         --site North --model_version 6.0.0 --bounds symmetric --padding 0.15

Plot layout from a parameter file with a list of telescopes:

.. code-block:: console

    simtools-plot-array-layout
        --array_layout_parameter_file tests/resources/model_parameters/array_layouts-2.0.2.json
        --model_version 6.0.0

Plot one layout from a parameter file:

.. code-block:: console

    simtools-plot-array-layout
        --array_layout_parameter_file tests/resources/model_parameters/array_layouts-2.0.2.json
        --array_layout_name_from_parameter_file alpha
        --model_version 6.0.0


Plot all layouts for the North site and model version 6.0.0:

.. code-block:: console

    simtools-plot-array-layout --site North --plot_all_layouts --model_version=6.0.0

Plot layout with some telescopes grayed out and others highlighted:

.. code-block:: console

    simtools-plot-array-layout --site North
                               --array_layout_name alpha
                               --model_version=6.0.0
                               --grayed_out_array_elements LSTN-01 LSTN-02
                               --highlighted_array_elements MSTN-01 MSTN-02
                               --legend_location "upper right"
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: plot_array_layout
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: plot_array_layout_by_name.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: plot_array_layout_from_array_layouts_json.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: plot_array_layout_from_list_north.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: plot_array_layout_from_list_south.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: plot_array_layout_from_list_utm_south.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: plot_array_layout_one_file.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: plot_array_layout_one_file_with_name.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: plot_array_layout_two_files.yml
```
