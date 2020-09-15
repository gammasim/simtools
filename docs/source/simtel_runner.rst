.. _simtel_runner:

sim_telarray Runner
===================

In this section you find the reference documentation of the sim_telarray runner module.
This module provides the main interface with the sim_telarray program provided by sim_telarray software.



The Ray Tracing module (and its main class RayTracing) handles ray tracing simulations, analysis and I/O. It receives as input
a Telescope Model (see :ref:`telescope_model`) and few more physical parameters (zenith angle, etc).
At the moment, the RayTracing class includes the whole dish mode (deafult one) and a single mirror mode.

The PSF Analysis module supports the Ray Tracing by handling a PSF Image. Its main class PSFImage
only receives the focal length and the list of photons positions. From that, it can handle the PSF calculation
and also the PSF image plotting.


.. automodule:: simtel_runner
   :members:
