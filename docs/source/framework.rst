.. _Framework:

Framework
=========

* `Configuration`_
* `IO`_


.. _Configuration:

Configuration
-------------

The configuration is handled by the :ref:`config <configmodule>` module (see reference documentation
below).

Configuration file
******************

gammasim-tools requires one configuration file in yaml format. This file should be named
config.yml and it should be located at the main gammasim-tools directory. Alternatively one can use
the function config.set_config_file_name to set an alternative file.

The configuration file must contain the following entries:

data_path: str
    Path to the directory containing the data used to perform automatic tests. These data is
    provided at gammasim-tools/data/.

output_path: str
    Path to the parent directory where the output directory will be created (see :ref:`I/O <IO>`).

model_path: list of str
    List with the locations of the model files. The locations of data from the instruments can be
    found on `gitlab (repo data-from-instrument-teams) <https://gitlab.cta-observatory.org/cta-
    consortium/aswg/simulations/simulation-model/verification/data-from-instrument-teams>`_. The
    location of the directory with the yaml files for the model parameters and the sim_telarray
    config files can be found on gitlab repos `configReports <https://gitlab.cta-observatory.org/
    cta-consortium/aswg/simulations/simulation-model/simulation-model-description/-/tree/master/
    configReports>`_ and `dataFiles <https://gitlab.cta-observatory.org/cta-consortium/aswg/
    simulations/simulation-model/simulation-model-description/-/tree/master/datFiles>`_.
    The files will always be searched recursively within all the subdirectories inside the directory
    given by the location.

simtel_path: str
    Path to the parent sim_telarray directory, containing the sim_telarray and CORSIKA software
    unpacked and compiled.

Environmental variables can be used to set the paths/locations. Example:


.. code-block:: console

    simtel_path: $SIMTELPATH

or

.. code-block:: console

    simtel_path: ${SIMTELPATH}


.. _configmodule:

config
******

.. automodule:: config
   :members:


.. _IO:

I/O
---

The I/O is handled by the :ref:`io_handler <iomodule>` module (see reference documentation below).

.. _iomodule:

io_handler
**********

.. automodule:: io_handler
   :members:
