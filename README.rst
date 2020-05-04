gammasim-tools
===========

MC Library for `CTA Observatory <www.cta-observatory.org>`_

Authors:
---------
    
* Raul R Prado (raul.prado@desy.de)
* Orel Gueta (orel.gueta@desy.de)


TODO:
------

* Fix the readthedocs (or set up any other web page). The html is succesfully imported to readthedocs but the automodule options are missing.    
* SimtelRunner: files (log, photon, star etc) names
* SimtelRunner: method to obtain script
* SimtelRunner: force flag
* Clean and organize SimtelRunner
* Corsika runner in SimtelRunner
* Docs in CorsikaConfig related code

GUIDELINES:
------------

* Follow `Style Guide and Code Guidelines from ctapipe <https://cta-observatory.github.io/ctapipe/development/index.html>`_
* Keep it clean!
* Names must be validated. (See util/names.py and its documentation)
* Sphinx for docs with docstrings in Google style (Should change to Numpy style, like ctapipe (?))
* Pep8
* Use the following namespaces consistently
.. code-block:: python
| import simtools.config as cfg
| import simtools.io_handler as io

* Bla

DESIGN REMARKS:
----------------

* Every functional class contains a 'label'. The label can be passed forward from lower level to higher level classes. In particular, label from TelescopeModel can be used in higher level classes.
* Classes are not designed to be re-used, all parameters should be set when initializing and not changed afterwards. New parameters should mean new instance of the class.
* filesLocation
* A test flag (test=True/False) should exist always when possible. If True, it must provide a faster and simpler implementation.
* Spectially important for time comsuming simulations. The default must always be test=False.
* A force flag (force=True/False) should exist in any case in which files are created. If False, the existing files should not be overwritten. The default must always be force=False.
* A clean method must be available for any class that create files.