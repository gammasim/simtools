.. _Framework:

Framework
=========

* `Configuration`_
* `IO`_
* `TestUnits`_


.. _Configuration:

Configuration
-------------

The configuration is handled by the :ref:`config <configmodule>` module (see reference documentation below).

Configuration file
******************

gammasim-tools requires one configuration file in yaml format. This file should be named
config.yml and it should be located at the local directory. Alternativelly one can use
the funtion config.setConfigFileName to set an alternative file. 

The configuration file must contain the following entries:

useMongoDB: bool
    Flag to turn on MongoDB. If False, the older yaml files will be used to access the Model Parameters.

mongoDBConfigFile: str
    Path to the MongoDB config file (see :ref:`Model Parameters DB`), which will be ignored if useMongoDB is False.

modelFilesLocations: list of str
    List with the locations of the model files.

outputLocation: str
    Path to the root directory where the output directory will be created (see :ref:`I/O <IO>`).

testDataLocation: str
    Path to the directory containing the data used to perform automatic tests. These data is provided at gammasim-tools/data/.

simtelPath: str
    Path to the root sim_telarray directory, containing the sim_telarray software unpacked and compiled.


Environmental variables can be used to set the paths/locations. Example:


.. code-block:: console

    simtelPath: $SIMTELPATH

or

.. code-block:: console

    simtelPath: ${SIMTELPATH}


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


.. _TestUnits:

Test Units
----------

Every library module must have a test unit associated at simtools/tests.
