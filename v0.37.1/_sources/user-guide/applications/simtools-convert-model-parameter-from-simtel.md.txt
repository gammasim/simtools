# simtools-convert-model-parameter-from-simtel

```{eval-rst}
.. automodule:: simtools.applications.convert_model_parameter_from_simtel
   :members:
   :exclude-members: main
```

```{eval-rst}
Check value, type, and range and write a json file ready to be submitted to the model database.

**Command line arguments**

parameter (str, required)
    Parameter name (as used in simtools)

simtel_cfg_file (str)
    File name of sim_telarray configuration file containing all simulation model parameters.

simtel_telescope_name (str)
    Name of the telescope in the sim_telarray configuration file.

telescope (str, optional)
    Telescope model name (e.g. LST-1, SST-D, ...)

**Example**

Extract the num_gains parameter from a sim_telarray configuration file for LSTN-01
and write a json file in the same format as the model parameter database:

.. code-block:: console

   simtools-convert-model-parameter-from-simtel \\
      --simtel_telescope_name CT1 \\
      --telescope LSTN-01 \\
      --schema_file tests/resources/num_gains.schema.yml \\
      --simtel_cfg_file tests/resources/simtel_config_test_la_palma.cfg \\
      --output_file num_gains.json
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: convert_model_parameter_from_simtel
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: convert_model_parameter_from_simtel_num_gains.yml
```
