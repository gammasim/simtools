# simtools-maintain-simulation-model-write-array-element-positions

```{eval-rst}
.. automodule:: simtools.applications.maintain_simulation_model_write_array_element_positions
   :members:
   :exclude-members: main
```

```{eval-rst}
This is an application for experts and should not be used by the general user.
Reading of input is fine-tuned to the array element files as provided by CTAO.

Writes one model parameter file per array element into a directory structure compatible
with the simtools model parameter repository.

Command line arguments

array_element_positions_file : str
    File containing a table of array element positions.
simulation_models_path : Path
    Path of local copy of model parameter repository.
parameter_version : str
    Parameter version.
coordinate_system : str
    Coordinate system of array element positions (ground or utm).

**Examples**
Add array element positions to repository (ground coordinates):

.. code-block:: console

    simtools-maintain-simulation-model-write-array-element-positions             --array_element_positions_file tests/resources/telescope_positions-North-ground.ecsv             --simulation_models_path /path/to/repository             --parameter_version 0.1.0             --coordinate_system ground

Add array element positions to repository (utm coordinates):

.. code-block:: console

    simtools-maintain-simulation-model-write-array-element-positions             --array_element_positions_file tests/resources/telescope_positions-North-utm.ecsv             --simulation_models_path /path/to/repository             --parameter_version 0.1.0             --coordinate_system utm
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: maintain_simulation_model_write_array_element_positions
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: maintain_simulation_model_write_array_element_positions_ground.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: maintain_simulation_model_write_array_element_positions_utm.yml
```
