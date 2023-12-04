.. _applications:


Applications
************

Applications are python scripts built on the :ref:`Library` that execute a well defined task.
Applications are the building blocks of `Simulation System Workflows <https://github.com/gammasim/workflows>`_.
Application scripts can be found in ``simtools/applications``.

Important: depending on the installation type, applications are named differently:

* developers (see :ref:`InstallationForDevelopers`) call applications as described throughout this documentation: ``python applications/<application name> ....``
* users (see :ref:`InstallationForUsers`) call applications directly as command line tool. Applications names ``simtools-<application name`` (with all ``_`` replaced by ``-``)

Each application is configured as described in :ref:`Configuration`.
The available arguments can be access by calling the ``python applications/<application name> --help``.

Some applications require one or multiple file names as input in the command line. The system will
first search on main simtools directory for these files, and in case it is not found, it will
search into the directories given by the config parameter *model_path*.

Output files of applications are written to $output_path/$label, where
*output_path* is a config parameter and *label* is the name of the application. The plots
produced directly by the application are stored in the sub-directory *application-plots*.
High-level data produced intermediately (e.g PSF tables) can be found in the sub-directories relative to
the specific type of application (e.g *ray-tracing* for optics related applications,
*camera-efficiency* for camera efficiency applications etc). All files related to the simulation model (e.g,
sim_telarray config files) are stored in the sub-directory *model*.


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

generate_corsika_histograms
===========================

.. automodule:: generate_corsika_histograms
   :members:

generate_default_metadata
=========================

.. automodule:: generate_default_metadata
   :members:

generate_simtel_array_histograms
================================

.. automodule:: generate_simtel_array_histograms
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

plot_array_layout
=================

.. automodule:: plot_array_layout
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


simulate_prod
=========================

.. automodule:: simulate_prod
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

validate_file_using_schema
==========================

.. automodule:: validate_file_using_schema
   :members:
