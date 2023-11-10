.. _Databases:

Databases
*********

The simtools package uses a prototype MongoDB database for the telescopes and sites model, reference data, derived values and test data.
Access to the DB is handeled via a dedicated API module. Access to the DB is restricted, please contact the developers in order to obtain access.

.. _Model Parameters DB:

Model Parameters DB
===================

The Model Parameters DB contains the parameters defining the telescopes in the simulation software. Logically, the DB is organised such that for each telescope, a list of parameters and their values is stored per version of the telescope model. Additional fields are stored as well, such units, value type, etc. Files relevant to the telescope model are directly saved in the DB and can be extracted using the file name.
