.. _simulators:

Simulators
==========

Base modules
------------

This modules the configuration, submission, and reporting for small simulation productions, used
e.g. for testing of different configurations.
Productions can be run locally, or submitted to a local batch farm (using gridengine or HTCondor).
Implemented are productions using CORSIKA and sim_telarray.

* :ref:`simulator <simulatormodule>`
* :ref:`job_manager <job_manager>`


.. _simulatormodule:

.. automodule:: simulator
   :members:


.. _job_manager:

.. automodule:: job_submission.job_manager
   :members:


Support modules for running CORSIKA
-----------------------------------

* :ref:`corsika_runner <corsika_runner>`
* :ref:`corsika_config <corsika_config>`

.. _corsika_runner:

.. automodule:: corsika.corsika_runner
   :members:

.. _corsika_config:

.. automodule:: corsika.corsika_config
   :members:


Support modules for running sim_telarray
----------------------------------------

The Ray Tracing module (and its main class RayTracing) handles ray tracing simulations, analysis and
I/O. It receives as input a Telescope Model (see :ref:`telescope_model`) and few more physical
parameters (zenith angle, etc). At the moment, the RayTracing class includes the whole dish mode
(default one) and a single mirror mode.

The PSF Analysis module supports the Ray Tracing by handling a PSF Image. Its main class PSFImage
only receives the focal length and the list of photons positions. From that, it can handle the PSF
calculation and also the PSF image plotting.

* :ref:`simtel_runner <simtel_runner>`
* :ref:`simtel_runner_array <simtel_runner_array>`
* :ref:`simtel_runner_ray_tracing <simtel_runner_ray_tracing>`
* :ref:`simtel_config_writer <simtel_config_writer>`
* :ref:`simtel_events <simtel_events>`
* :ref:`simtel_histograms <simtel_histograms>`

.. _simtel_runner:

.. automodule:: simtel.simtel_runner
   :members:

.. _simtel_runner_array:

.. automodule:: simtel.simtel_runner_array
   :members:

.. _simtel_runner_ray_tracing:

.. automodule:: simtel.simtel_runner_ray_tracing
   :members:

.. _simtel_config_writer:

.. automodule:: simtel.simtel_config_writer
   :members:

.. _simtel_events:

.. automodule:: simtel.simtel_events
   :members:

.. _simtel_histograms:

.. automodule:: simtel.simtel_histograms
   :members:
