# simtools-convert-geo-coordinates-of-array-elements

```{eval-rst}
.. automodule:: simtools.applications.convert_geo_coordinates_of_array_elements
   :members:
   :exclude-members: main
```

```{eval-rst}
**Description**

Convert array element positions in different CTAO coordinate systems.
Available coordinate systems are:

1. UTM system
2. ground system (similar to sim_telarray system with x-axis pointing toward geographic north
   and y-axis pointing towards the west); altitude relative to the CORSIKA observation level.
   Altitude is the height of the elevation rotation axis (plus some possible mirror offset).
3. Mercator system

**Command line arguments**
array_element_positions_file (str)
    File name with list of array element positions.
    Input can be given as astropy table file (ecsv) or a single array element in
    a json file.
print (str)
    Print in requested coordinate system; possible are ground, utm, mercator
export (str)
    Export array element list to file in requested coordinate system;
      possible are ground, utm, mercator
select_assets (str)
    Select a subset of array elements / telescopes (e.g., MSTN, LSTN)

**Example**
Convert a list of array elements using a list of telescope positions in UTM coordinates.

.. code-block:: console

    simtools-convert-geo-coordinates-of-array-elements
        --array_element_positions_file tests/resources/telescope_positions-North-utm.ecsv
        --print ground

The converted list of telescope positions in ground coordinates is printed to the screen.

The following example converts a list of telescope positions in UTM coordinates
and writes the output to a file in ground (sim_telarray) coordinates. Also selects
only a subset of the array elements (telescopes; ignore calibration devices):

.. code-block:: console

    simtools-convert-geo-coordinates-of-array-elements
        --array_element_positions_file tests/resources/telescope_positions-North-utm.ecsv
        --export ground
        --select_assets LSTN

Expected output is a ecsv file in the directory printed to the screen.
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: convert_geo_coordinates_of_array_elements
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: convert_geo_coordinates_of_array_elements_ground_to_ground.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: convert_geo_coordinates_of_array_elements_ground_to_utm_json.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: convert_geo_coordinates_of_array_elements_mercator_to_utm_meta_in_table.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: convert_geo_coordinates_of_array_elements_print_compact_corsika_telescopeheights.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: convert_geo_coordinates_of_array_elements_print_compact_nocors_corsika.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: convert_geo_coordinates_of_array_elements_print_compact_nocors_utm.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: convert_geo_coordinates_of_array_elements_print_ground.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: convert_geo_coordinates_of_array_elements_utm_to_ground_json.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: convert_geo_coordinates_of_array_elements_utm_to_ground_meta_in_table.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: convert_geo_coordinates_of_array_elements_utm_to_ground_meta_in_yml.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: convert_geo_coordinates_of_array_elements_utm_to_mercator_meta_in_table.yml
```
