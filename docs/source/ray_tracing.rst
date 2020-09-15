.. _RayTracing:

Ray Tracing
===========

In this section you find the reference documentation of the Ray Tracing and  PSF Analysis modules.
The Ray Tracing module (and its main class RayTracing) handles ray tracing simulations, analysis and I/O. It receives as input
a Telescope Model (see :ref:`telescope_model`) and few more physical parameters (zenith angle, etc).
At the moment, the RayTracing class includes the whole dish mode (deafult one) and a single mirror mode.

The PSF Analysis module supports the Ray Tracing by handling a PSF Image. Its main class PSFImage
only receives the focal length and the list of photons positions. From that, it can handle the PSF calculation
and also the PSF image plotting.

* `ray_tracing <raytracingmodule>`_
* `psf_analysis <psfanalysismodule>`_


.. _raytracingmodule:

ray_tracing
-----------

.. automodule:: ray_tracing
   :members:


.. _psfanalysismodule:

psf_analysis
------------

.. automodule:: psf_analysis
   :members:
