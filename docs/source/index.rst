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

gammasim-tools is a prototype of a software package that provides tools and
the framework for the Simulation System of the `CTA Observatory <www.cta-observatory.org>`_

Among the main functionalities of gammasim-tools, there are:

* simulation model DB interface and management;
* simulation model parameter derivation and validation.

The design principles followed by gammasim-tools are:

* standardized interface and data products independent of the underlying software (e.g., CORSIKA, sim_telarray, GrOPTICS);
* maximizes the sharing of tools and algorithms with other DPPS subsystems (e.g., `ctapipe <https://github.com/cta-observatory/ctapipe>`_ and calibpipe);
* I/O and reporting tools for the MC simulation model parameter and production database.

Contact
=======
    
* Raul R Prado (raul.prado@desy.de)
* Orel Gueta (orel.gueta@desy.de)
* Gernot Maier (gernot.maier@desy.de)


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