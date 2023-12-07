==============
simtools
==============

.. image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
   :target: https://github.com/gammasim/simtools/blob/main/LICENSE

.. image:: https://img.shields.io/github/v/release/gammasim/simtools
   :target: https://github.com/gammasim/simtools/releases

.. image:: https://zenodo.org/badge/195011575.svg
   :target: https://zenodo.org/badge/latestdoi/195011575

.. image:: https://badge.fury.io/py/gammasimtools.svg
    :target: https://badge.fury.io/py/gammasimtools

.. image:: https://github.com/gammasim/simtools/actions/workflows/CI-unittests.yml/badge.svg
   :target: https://github.com/gammasim/simtools/actions/workflows/CI-unittests.yml

.. image:: https://github.com/gammasim/simtools/actions/workflows/CI-integrationtests.yml/badge.svg
   :target: https://github.com/gammasim/simtools/actions/workflows/CI-integrationtests.yml

.. image:: https://github.com/gammasim/simtools/actions/workflows/CI-docs.yml/badge.svg
   :target: https://github.com/gammasim/simtools/actions/workflows/CI-docs.yml

.. image:: https://app.codacy.com/project/badge/Grade/a3f19df7454844059341edd0769e02a7
   :target: https://app.codacy.com/gh/gammasim/simtools/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade

.. image:: https://codecov.io/gh/gammasim/simtools/graph/badge.svg?token=AYAIRPARCH
   :target: https://codecov.io/gh/gammasim/simtools

Simulation tools and applications for the `Cherenkov Telescope Array (CTAO) <https://www.cta-observatory.org>`_.

License: `BSD-3 <https://github.com/gammasim/simtools/blob/main/LICENSE>`_

Useful links:
`Documentation <https://gammasim.github.io/simtools/>`_
`Source code <https://github.com/gammasim/simtools>`_
`Issue tracker <https://github.com/gammasim/simtools/issues>`_
`Citation <https://github.com/gammasim/simtools/blob/main/CITATION.cff>`_

simtools provides:

* libraries for simulation model management and model database interface
* tools for the preparation and configuration of simulation productions
* applications for simulation model parameter derivation and validation
* standardized interfaces and data products independent of the underlying simulation software (e.g., `CORSIKA <https://www.iap.kit.edu/corsika/>`_, `sim_telarray <https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray/>`_)
* I/O and reporting tools for the MC simulation model parameter and production database

simtools is one part of the CTAO Simulation Pipeline, which consist of the following components:

- `CORSIKA <https://www.iap.kit.edu/corsika/>`_ air shower simulation code and the `sim_telarray <https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray/>`_ telescope simulation code
- `workflows <https://github.com/gammasim/workflows>`_ for setting, derivation and validation of simulation model parameters
- `simulation model parameter and input data schema <https://github.com/gammasim/workflows/tree/main/schemas>`_
- `databases <https://gammasim.github.io/simtools/databases.html>`_, especially the model parameter database

simtools is under rapid development with continuous changes and additions planned.
Please contact the developers before using it: simtools-developer@desy.de


Quickstart
==========

.. warning::

    The pip-installation of simtools provides limited functionality only
    and is not as well tested as the conda/mamba installation.

Install simtools with pypi (recommended for users):

.. code-block:: bash

    pip install gammasimtools

Install simtools with mamba (recommended for developers):

.. code-block:: bash

    git clone https://github.com/gammasim/simtools.git
    cd simtools
    mamba env create -f environment.yml
    conda activate simtools
    pip install -e .

For more detail see the `Getting_Started <https://gammasim.github.io/simtools/getting_started.html>`_ section of the manual.

Authors
=======

* Raul R Prado (DESY)
* Orel Gueta (DESY)
* Tobias Kleiner (DESY)
* Victor B. Martins (DESY)
* Gernot Maier (DESY)

Citing this Software
====================

Please cite this software if it use used for a publication, see the `Zenodo record <https://doi.org/10.5281/zenodo.6346696>`_ and `CITATION.cff <https://github.com/gammasim/simtools/blob/main/CITATION.cff>`_ .

Acknowledgements
================

This project is supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - project number 460248186 (PUNCH4NFDI).
