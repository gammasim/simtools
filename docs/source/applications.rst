.. _applications:


Applications
************

Introduction
============

Applications are python scripts built on the :ref:`Library` that execute a well defined task.
These applications will be used as building blocks for the Simulation System Workflows.

The application scripts can be found in gammasim-tools/applications.

The applications expect a config file named config.yml in your gammasim-tools directory (unless
specified differently using the function config.set_config_file_name) . Please, follow the instructions
in :ref:`Configuration` to setup your config file. Alternatively, you can set up environmental
variables through the *set_env_db_template.sh* script or pass it as arguments to the applications.
The available arguments can be access by calling *--help* after calling the application.

Some applications require one or multiple file names as input in the command line. The system will
first search on main gammasim-tools directory for these files, and in case it is not found, it will
search into the directories given by the config parameter *model_path*.

The output files from the applications will be written to $output_path/$label, where
*output_path* is a config parameter and *label* is the name of the application. The plots
produced directly by the application are stored in the sub-directory *application-plots*. The high-
level data produced intermediately (e.g PSF tables) can be found in the sub-directories relative to
the specific type of application (e.g *ray-tracing* for optics related applications,
*camera-efficiency* for camera efficiency applications etc). All files related to the model (e.g
sim_telarray config files) are stored in the sub-directory *model*.

Below you find the list of the currently available applications and their respective documentation.

List of Applications
====================

* `add_file_to_db`_
* `compare_cumulative_psf`_
* `derive_mirror_rnda`_
* `get_file_from_db`_
* `get_parameter`_
* `make_regular_arrays`_
* `plot_simtel_histograms`_
* `print_array_elements`_
* `produce_array_config`_
* `production`_
* `sim_showers_for_trigger_rates`_
* `submit_data_from_external`_
* `tune_psf`_
* `validate_camera_efficiency`_
* `validate_camera_fov`_
* `validate_optics`_


add_file_to_db
==============

.. automodule:: add_file_to_db
   :members:


compare_cumulative_psf
======================

.. automodule:: compare_cumulative_psf
   :members:


derive_mirror_rnda
==================

.. automodule:: derive_mirror_rnda
   :members:

get_file_from_db
================

.. automodule:: get_file_from_db
   :members:

get_parameter
=============

.. automodule:: get_parameter
   :members:


make_regular_arrays
===================

.. automodule:: make_regular_arrays
   :members:


mark_non_optics_parameter_non_applicable
========================================

.. automodule:: mark_non_optics_parameter_non_applicable
   :members:


plot_simtel_histograms
======================

.. automodule:: plot_simtel_histograms
   :members:


print_array_elements
====================

.. automodule:: print_array_elements
   :members:


produce_array_config
====================

.. automodule:: produce_array_config
   :members:


production
==========

.. automodule:: production
   :members:


sim_showers_for_trigger_rates
=============================
.. automodule:: sim_showers_for_trigger_rates
   :members:


submit_data_from_external
=========================

.. automodule:: submit_data_from_external
   :members:


tune_psf
========

.. automodule:: tune_psf
   :members:


validate_camera_efficiency
==========================

.. automodule:: validate_camera_efficiency
   :members:


validate_camera_fov
===================

.. automodule:: validate_camera_fov
   :members:


validate_optics
===============

.. automodule:: validate_optics
   :members:
