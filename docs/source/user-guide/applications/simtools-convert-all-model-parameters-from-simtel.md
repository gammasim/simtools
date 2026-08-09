# simtools-convert-all-model-parameters-from-simtel

```{eval-rst}
.. automodule:: simtools.applications.convert_all_model_parameters_from_simtel
   :members:
   :exclude-members: main
```

```{eval-rst}
Check value, type, and range, convert units using schema files. Write json files
ready to be submitted to the model database. Prints out parameters which are not found
in sim_telarray configuration file and parameters which are not found in simtools schema files.

Note that all parameters are assigned the same parameter version.

**Command line arguments**
simtel_cfg_file (str)
    File name of sim_telarray configuration file containing all simulation model parameters.

simtel_telescope_name (str)
    Name of the telescope in the sim_telarray configuration file.

telescope (str, optional)
    Telescope model name (e.g. LST-1, SST-D, ...)

skip_parameter (str, optional)
    List of parameters to be skipped (use sim_telarray names).

**Example**

To export the model parameters from sim_telarray, first copy and unpack the configuration
tar ball from sim_telarray (usually called 'sim_telarray_config.tar.gz') to the sim_telarray
working directory. Extract the configuration using the following command:

.. code-block:: console

    ./sim_telarray/bin/sim_telarray -c sim_telarray/cfg/CTA/CTA-PROD6-LaPalma.cfg \\
        -C limits=no-internal -C initlist=no-internal -C list=no-internal \\
        -C typelist=no-internal -C maximum_telescopes=30 -DNSB_AUTOSCALE \\
        -DNECTARCAM -DHYPER_LAYOUT -DNUM_TELESCOPES=30 /dev/null \\
        2>|/dev/null | grep '(@cfg)' | sed 's/^(@cfg) //' >| all_telescope_config_la_palma.cfg

    ./sim_telarray/bin/sim_telarray -c sim_telarray/cfg/CTA/CTA-PROD6-Paranal.cfg \\
        -C limits=no-internal -C initlist=no-internal -C list=no-internal \\
        -C typelist=no-internal -C maximum_telescopes=87 -DNSB_AUTOSCALE \\
        -DFLASHCAM -DHYPER_LAYOUT -DNUM_TELESCOPES=87 /dev/null \\
        2>|/dev/null | grep '(@cfg)' | sed 's/^(@cfg) //' >| all_telescope_config_paranal.cfg


Extract then model parameters from the sim_telarray configuration file for LSTN-01
(telescopes are named CT1, CT2, ..., in the sim_telarray configuration file and must be
provided in the "simtel_telescope_name" command line argument)
and write json files in the same format as the model parameter database:

.. code-block:: console

   simtools-convert-all-model-parameters-from-simtel \\
      --simtel_cfg_file all_telescope_config_la_palma.cfg\\
      --simtel_telescope_name CT1\\
      --telescope LSTN-01\\
      --parameter_version "1.0.0"\\
      --output_path /path/to/output
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: convert_all_model_parameters_from_simtel
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: convert_all_model_parameters_from_simtel_num_gains.yml
```
