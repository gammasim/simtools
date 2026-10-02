# simtools-generate-regular-arrays

```{eval-rst}
.. automodule:: simtools.applications.generate_regular_arrays
   :members:
   :exclude-members: main
```

```{eval-rst}
Arrays can consist of single (central) telescopes, square grids or star-like (with
telescopes arranged on main axes) patterns.  All telescopes in the array are of
the same type and are placed at regular distances.

Output files are saved as astropy tables in ASCII ECSV format and in the simtools format
required to be used for the overwrite model parameter configuration.

**Command line arguments**

telescope_type (str)
    Type of telescope (e.g., LST, MST, SST).
number_of_telescopes (int)
    Number of telescopes in the array.
telescope_distance (float)
    Distance between telescopes in the array (in meters).
array_shape (str)
    Shape of the array ('square', 'star').
site (str, required)
    observatory site (e.g., North or South).
model_version (str, optional)
    Model version to use (e.g., 6.0.0). If not provided, the latest version is used.

**Example**

Runtime < 10 s.

.. code-block:: console

    simtools-generate-regular-arrays --site=North
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: generate_regular_arrays
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: generate_regular_arrays_run_north_40mst_star.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: generate_regular_arrays_run_south_1sst.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: generate_regular_arrays_run_south_4lst_square.yml
```
