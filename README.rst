gammasim-tools
===========

.. image:: https://zenodo.org/badge/195011575.svg
   :target: https://zenodo.org/badge/latestdoi/195011575

.. image:: https://github.com/gammasim/gammasim-tools/actions/workflows/CI-tests.yml/badge.svg
   :target: https://github.com/gammasim/gammasim-tools/actions/workflows/CI-tests.yml

.. image:: https://github.com/gammasim/gammasim-tools/actions/workflows/CI-docs.yml/badge.svg
   :target: https://github.com/gammasim/gammasim-tools/actions/workflows/CI-docs.yml

.. image:: https://codecov.io/gh/gammasim/gammasim-tools/branch/master/graph/badge.svg?token=AYAIRPARCH
   :target: https://codecov.io/gh/gammasim/gammasim-tools

Prototype implementation of tools for the Simulation System of the `CTA Observatory <www.cta-observatory.org>`_

gammasim-tools provides a framework and tools for:

* simulation model DB interface and management
* simulation model parameter derivation and validation

gammasim-tools follows these design principles:

* standardized interface and data products independent of the underlying software (e.g., CORSIKA, sim_telarray, GrOPTICS)
* maximizes the sharing of tools and algorithms with other DPPS subsystems (e.g., `ctapipe <https://github.com/cta-observatory/ctapipe>`_ and calibpipe)
* I/O and reporting tools for the MC simulation model parameter and production database

gammasim-tools makes extensive use of the `CORSIKA <https://www.iap.kit.edu/corsika/>`_ air shower simulation code and the `sim_telarray <https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray/>`_ telescope simulation code.

AUTHORS:
--------

* Raul R Prado (raul.prado@desy.de)
* Orel Gueta (orel.gueta@desy.de)
* Gernot Maier (gernot.maier@desy.de)

INSTALLATION:
-------------

Create a conda environment (only once):

``conda env create -f environment.yml``

``conda activate gammasim-tools-dev``

``pip install -e .``

Each time you want to use the package, just activate the conda environment:

``conda activate gammasim-tools-dev``

STYLE GUIDELINES:
-----------------

* Follow `Style Guide and Code Guidelines from ctapipe <https://cta-observatory.github.io/ctapipe/development/index.html>`_
* Keep it clean!
* Sphinx for docs with docstrings in `Numpy style <https://numpydoc.readthedocs.io/en/latest/format.html#id4>`_
* Pep8 is required.
