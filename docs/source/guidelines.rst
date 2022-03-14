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


Telescope Names
---------------

The telescope names as used by gammasim-tools follow the pattern "Site-Class-Type", where:

* "Site" is either "North" or "South";
* "Class" is either "LST", "MST", "SCT" or "SST";
* "Type" is a single number ONLY in case of a real telescope existing at the site or a string containing a "D" in case of any other telescope design.

For example:

* "North-LST-1" is the first LST commissioned at the La Palma site, while "North-LST-D234" is the current design of the further 3 LSTs.
* "North-MST-FlashCam-D" and "North-MST-NectarCam-D" are the two MST designs containing different cameras.

Any input telescope names can (and should) be validated by the function validateTelescopeName (see :ref:`util.names <utilnames>`).
For the Site field, any different capitalization (e.g "south") or site names like "paranal" and "lapalma" will be accepted
and converted to the standard ones. The same applies to the Class field.
For the Type field, any string will be accepted and a selected list of variations will be converted to the standard ones
(e.g "flashcam" will be converted to "FlashCam").


Validating names
================

Any name that is reccurently used along the the package should be validated when given as input.
Examples of names are: telescope, site, camera, model version. The functionaties to validate names
are found in util.names. The function validateName receives the input string and a name dictionary,
that is usually called allSomethingNames. This dictionary contain the possible names (as keys) and lists
of allowed alternatives names as values. In case the input name is found in one of the lists, the key
is returned.

The name dictnaries are also defined in util.names. One should also define especific functions named
validateSomethingNames that call the validateName with the proper name dictionary. This is only meant to
provide a clear interface.

This is an example of a name dictionary:


.. code-block:: yaml

  allSiteNames = {
    "South": ["paranal", "south"],
    "North": ["lapalma", "north"]
  }

And this is an example of how the site name is validated in the telescope model module:


.. code-block:: python

  self.site = names.validateSiteName(site)

where site was given as parameter to the __init__ function.


Handling data files
===================

Data files are auxiliary files containing data required to run simtools.
These day is kept in files to avoid having it hardcoded throughout the code.
They are located in the data directory inside the main repository and its location
must be given in the config file.

Data files are organized in subdirectories. One can get a data file using the function
io.getDataFile(subdirectory, filename).

Examples of files that are kept as data files are: test files, ecsv files used to define
the layouts and parameter files (see Input validation section).


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

.. code-block:: yaml

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
* names is a list of acceptable input names. The key in the returned dict will have the name given at the definition of the block (zenithAngle in this example)


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
