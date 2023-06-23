==============
simtools
==============

.. image:: https://zenodo.org/badge/195011575.svg
   :target: https://zenodo.org/badge/latestdoi/195011575

.. image:: https://github.com/gammasim/simtools/actions/workflows/CI-unittests.yml/badge.svg
   :target: https://github.com/gammasim/simtools/actions/workflows/CI-unittests.yml

.. image:: https://github.com/gammasim/simtools/actions/workflows/CI-integrationtests.yml/badge.svg
   :target: https://github.com/gammasim/simtools/actions/workflows/CI-integrationtests.yml

.. image:: https://github.com/gammasim/simtools/actions/workflows/CI-docs.yml/badge.svg
   :target: https://github.com/gammasim/simtools/actions/workflows/CI-docs.yml

.. image:: https://app.codacy.com/project/badge/Grade/717d4dc06dfa45d3ad0d61499ffc0e2e
   :target: https://www.codacy.com/gh/gammasim/simtools/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=gammasim/gammasim-tools&amp;utm_campaign=Badge_Grade

.. image:: https://codecov.io/gh/gammasim/simtools/branch/master/graph/badge.svg?token=AYAIRPARCH
   :target: https://codecov.io/gh/gammasim/simtools


Prototype implementation of tools for the Simulation Pipeline of the `CTA Observatory <www.cta-observatory.org>`_:

* libraries for simulation model management and model database interface;
* applications for simulation model parameter derivation and validation;
* standardized interfaces and data products independent of the underlying simulation software (e.g., `CORSIKA <https://www.iap.kit.edu/corsika/>`_, `sim_telarray <https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray/>`_);
* I/O and reporting tools for the MC simulation model parameter and production database.

simtools is one part of the CTAO Simulation Pipeline, which consist of the following components:

- `CORSIKA <https://www.iap.kit.edu/corsika/>`_ air shower simulation code and the `sim_telarray <https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray/>`_ telescope simulation code.
- simtools for tools and application for model parameter and configuration management, database interfaces, validation, and production preparations.
- databases, especially the model parameter database.

This code is under rapid development. Please contact the developers if you want to use it.

- Code: https://github.com/gammasim/simtools
- Documentation: https://gammasim.github.io/simtools/
- Model database access: contact developers.

Authors
=======

* Raul R Prado (raul.prado@desy.de)
* Orel Gueta (orel.gueta@desy.de)
* Victor B. Martins (victor.barbosa.martins@desy.de)
* Gernot Maier (gernot.maier@desy.de)

Citing this Software
====================

Please cite this software if you use it for a publication.
Please cite the `Zenodo record <https://doi.org/10.5281/zenodo.6346696>`_, see the `CITATION.cff <https://github.com/gammasim/simtools/blob/master/CITATION.cff>`_ file.

INSTALLATION:
=============

Follow the steps outlined in the `Getting Started <https://gammasim.github.io/simtools/getting_started.html>`_ of the manual.
