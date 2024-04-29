.. _AUXILIARYFILES:

Auxiliary Files
***************

Auxiliary parameter files include example configurations and array layouts:

1. `data/parameters <https://github.com/gammasim/simtools/tree/main/data/parameters/>`_: parameter type and unit definitions (e.g., defining the units for zenith angle for configuration parameters for CORSIKA)
2. `data/layout <https://github.com/gammasim/simtools/tree/main/data/layout/>`_: layout definitions for single telescope or simple 4-telescope grid layouts (used e.g., for trigger simulations)

Data files (e.g., quantum efficiency tables or camera trigger definitions) are stored in the simulation model database, see :ref:`Model Parameters DB`.

Integration and unit tests provide a rich set of examples for the usage of the simulation tools.
The configuration and data files located in `tests/resources <https://github.com/gammasim/simtools/tree/main/tests/resources/>`_ are used for the tests.
