.. _applications:


Applications
************

Introduction
============

Applications are python scripts built on the :ref:`Library` that execute a simple, well defined task.
These applications will be used as building blocks for the Simulation System Workflows.

The application scripts can be found in gammasim-tools/applications.

The applications expect a config file named config.yml in your running directory. Please,
follow the instructions in :ref:`Configuration` to setup your config file.

Some applications require one or multiple file names as input in the command line. The system will
first search on your local directory for these files, and in case it is not found, it will search
into the directories given by the config parameter *modelFilesLocation*.

The output files from the applications will be written to $outputLocation/$label, where *outputLocation*
is a config parameter and *label* is the name of the application. The plots produced directly by the application
are stored in the sub-directory *application-plots*. The high-level data produced intermediately (e.g PSF tables)
can be found in the sub-directories relative to the specific type of application (e.g *ray-tracing* for optics
related applications, *camera-efficincy* for camera efficiency applications etc). All files related to the
model (e.g sim_telarray config files) are stored in the sub-directory *model*.    

Below you find the list of the currently available applications and their respective documentation.

List of Applications
====================

* `derive_mirror_rnda`_
* `validate_optics`_
* `compare_cumulative_psf`_
* `validate_camera_efficiency`_
* `validate_camera_fov`_
* `make_regular_arrays`_
* `produce_array_config`_
* `sim_showers_for_trigger_rates`_


derive_mirror_rnda
------------------

.. automodule:: derive_mirror_rnda
   :members:


validate_optics
---------------

.. automodule:: validate_optics
   :members:


compare_cumulative_psf
----------------------


.. automodule:: compare_cumulative_psf
   :members:



validate_camera_efficiency
--------------------------

.. automodule:: validate_camera_efficiency
   :members:


validate_camera_fov
-------------------

.. automodule:: validate_camera_fov
   :members:


make_regular_arrays
-------------------

.. automodule:: make_regular_arrays
   :members:


produce_array_config
--------------------

.. automodule:: produce_array_config
   :members:

sim_showers_for_trigger_rates
-----------------------------

.. automodule:: sim_showers_for_trigger_rates
   :members:
