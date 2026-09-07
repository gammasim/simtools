# simtools-generate-array-config

```{eval-rst}
.. automodule:: simtools.applications.generate_array_config
   :members:
   :exclude-members: main
```

```{eval-rst}
The applications generates the sim_telarray configuration files for a given array, site,
and model_version using the model parameters stored in the database.

**Command line arguments**

site : str
    Site name (e.g., North, South).
array_layout_name : str
    Name of the layout array (e.g., test_layout, alpha, 4mst, etc.).
array_element_list : list
    List of array elements (e.g., telescopes) to plot (e.g., ``LSTN-01 LSTN-02 MSTN``).

**Example**

North - 5.0.0:

.. code-block:: console

    simtools-generate-array-config --site North --array_layout_name alpha --model_version 5.0.0

The output is saved in simtools-output/test/model.
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: generate_array_config
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: generate_array_config_run.yml
```
