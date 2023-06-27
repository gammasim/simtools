.. _SimulationSoftware:

Simulation Software
*******************

Simulation software is external to simtools and used by several applications.
This includes the main software tools CORSIKA and sim_telarray, and several other tools listed in the following

CORSIKA
-------

The following executable are called in simtools from the `CORSIKA <https://www.iap.kit.edu/corsika/>`_ package:

* ``corsika``: the CORSIKA air shower simulation code

sim_telarray
------------

The following executable are called in simtools from the `sim_telarray <https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray>`_ package:

* ``sim_telarray``: the main ray-tracing and detector simulation code
* ``testeff``: telescope and camera efficiency calculation
* ``rx``: optical PSF calculation (optional use)
* ``corsika_autoinputs``: generate CORSIKA input parameters
* ``pfp``: sim_telarray file pre processor
