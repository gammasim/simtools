.. _Guidelines:

Guidelines for Developers
*************************

This section is meant for developers.


Conventions
===========

Imports
-------

import simtools.config as cfg
import simtools.util.general as gen
import simtools.io_handler as io


Handling data files
===================


Input validation
================

Any module that receives configurable inputs (e.g. physical parameters)
must have them validated. The validation assures that the units, type and
format are correct and also allow for default values.

The configurable input must be passed to classes through a dictonary or a yaml
file. In the case of a dictionary the parameter is called configData, and in the
case of a yaml file, configFile. See the ray_tracing module for an example.

The function gen.collectDataFromYamlOrDict(configData, configFile, allowEmpty=False)
must be used to read these arguments. It identifies which case was given and
reads it accordinly, returnig a dictinary. It also raises an exception in case none are
given and not allowEmpty.

The validation of the input is done by the function gen.validateConfigData, which
receives the dictionary with the collected input and a parameter dictionary. The parameter 
dictionary is read from a parameter yaml file in the data/parameters directory.
The file is read through the function io.getDataFile("parameters", filename)
(see data files section). 

The parameter yaml file contains the list of parameters to be validated and its
properties. See an example below:

.. code-block::

zenithAngle: 
  len: 1
  unit: !astropy.units.Unit {unit: deg}
  default: !astropy.units.Quantity
    value: 20
    unit: !astropy.units.Unit {unit: deg}
  names: ['zenith', 'theta']


* len gives the length of the input. If null, any len is accepted.
* unit is the astropy unit
* default must have the same len
* names is a list of acceptable input names. The key in the returned dict will have the name given
at the definition of the block (zenithAngle in this example)


Testing
=======

pytest framework is used for unit testing.
The test modules are located in simtools/test.
Every module should have its respective test module and
ideally all functions should be covered by tests.

It is important to write the tests in parallel with the modules
to assure that the code is testable.

The pytest decorators mark.ignoreif are used to mark the tests that
requires: a) a config file properly set, b) a sim_telarray installation and
c) DB connection. Each of these are identified before each pytest session
and environment variables are used to store this information. See the implementation
in conftest.py. In util/tests.py one can find functions that reads these variables.
