# simtools-db-get-array-layouts-from-db

```{eval-rst}
.. automodule:: simtools.applications.db_get_array_layouts_from_db
   :members:
   :exclude-members: main
```

```{eval-rst}
To get the list of pre-defined array layouts, use ``--list_available_layouts``.

To get the list of array elements for a given layout, use ``--array_layout_name``.

To get the positions for a set of array elements, use ``--array_element_list``.
Listing of array elements follows this logic:

* explicit listing: e.g., ``-array_element_list MSTN-01, MSTN05``
* listing of types: e.g, ``-array_element_list MSTN`` plots all telescopes of type MSTN.

**Command line arguments**
list_available_layouts : bool, optional
    List available layouts in the database.
include_calibration_array_elements : bool, optional
    Include calibration array elements in output table (default: only telescopes).
array_layout_name : str
    Name of the layout array (e.g., test_layout, alpha, 4mst, etc.).
array_element_list : list
    List of array elements (e.g., telescopes) to plot (e.g., ``LSTN-01 LSTN-02 MSTN``).
coordinate_system : str, optional
    Coordinate system for the array layout (ground or utm).
output_file : str, optional
    Name of the output file to be saved as astropy table (ecsv file)

**Examples**
List pre-defined array layouts.

.. code-block:: console

    simtools-db-get-array-layouts-from-db --site North --model_version "6.0.0"

Retrieve telescope positions for array layout 'test_layout' from database.

.. code-block:: console

    simtools-db-get-array-layouts-from-db --site North --model_version "6.0.0"
        --array_layout_name test_layout

Retrieve telescope positions from database (utm coordinate system) and write to an ecsv files

.. code-block:: console

    simtools-db-get-array-layouts-from-db --site North --model_version "6.0.0"
      --array_element_list LSTN-01 LSTN-02 MSTN
      --coordinate_system utm
      --output_file telescope_positions-test_layout.ecsv

Retrieve array-element positions including calibration elements.

.. code-block:: console

        simtools-db-get-array-layouts-from-db --site South --model_version "6.0.2"
            --array_element_list LSTS ILLS
            --include_calibration_array_elements
            --output_file array_layout_south_ground_with_calibration.ecsv
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: db_get_array_layouts_from_db
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: db_get_array_layouts_from_db_layout_list.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: db_get_array_layouts_from_db_layout_name.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: db_get_array_layouts_from_db_layout_with_calibration_flag.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: db_get_array_layouts_from_db_list_arrays.yml
```
