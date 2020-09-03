.. ctamclib documentation master file, created by
   sphinx-quickstart on Mon Jul  8 11:04:43 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _gammasimtools:

==============
gammasim-tools
==============


What is gammasim-tools?
=======================

Prototype implementation of tools for the Simulation System of the `CTA Observatory <www.cta-observatory.org>`_

gammasim-tools provides a framework and tools for:

* simulation model DB interface and management
* simulation model parameter derivation and validation

gammasim-tools follows these design principles:

* standardized interface and data products independent of the underlying software (e.g., CORSIKA, sim_telarray, GrOPTICS)
* maximizes the sharing of tools and algorithms with other DPPS subsystems (e.g., `ctapipe <https://github.com/cta-observatory/ctapipe>`_ and calibpipe)
* I/O and reporting tools for the MC simulation model parameter and production database

Contact
=======
    
* Raul R Prado (raul.prado@desy.de)
* Orel Gueta (orel.gueta@desy.de)


General Documentation
=====================

.. toctree::
  :maxdepth: 2
  :glob:

  getting_started
  applications
  framework
  library
  databases