.. _gammasimtools:

==============
gammasim-tools
==============


What is gammasim-tools?
=======================

gammasim-tools is a prototype software package that provides tools and
the framework for the Simulation System of the `CTA Observatory <https://www.cta-observatory.org/>`_

Among the main functionalities of gammasim-tools are:

* simulation model DB interface and management;
* simulation model parameter derivation and validation.

The design principles followed by gammasim-tools are:

* standardized interface and data products independent of the underlying software (e.g., CORSIKA, sim_telarray, GrOPTICS);
* maximizes the sharing of tools and algorithms with other DPPS subsystems (e.g., `ctapipe <https://github.com/cta-observatory/ctapipe>`_ and calibpipe);
* I/O and reporting tools for the MC simulation model parameter and production database.

Authors
=======

|author|

Citation
========

See citation file (`CITATION.cff <https://github.com/gammasim/gammasim-tools/blob/master/CITATION.cff>`_) on how to site this software.


General Documentation
=====================

.. toctree::
  :maxdepth: 2
  :glob:

  getting_started
  framework
  applications
  library
  databases
  guidelines
