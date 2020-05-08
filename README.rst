gammasim-tools
===========

Core library and tools for the Simulation System of `CTA Observatory <www.cta-observatory.org>`_


AUTHORS:
--------
    
* Raul R Prado (raul.prado@desy.de)
* Orel Gueta (orel.gueta@desy.de)

INSTALLATION:
-------------

Create (only once) and activate your conda enviroment:

conda env create -f environment.yml
conda activate gammasim-tools-dev

While a proper conda installation has not been setup, you can add simtools to your python path by

source set_simtools.sh
 
STYLE GUIDELINES:
-----------------

* Follow `Style Guide and Code Guidelines from ctapipe <https://cta-observatory.github.io/ctapipe/development/index.html>`_
* Keep it clean!
* Sphinx for docs with docstrings in `Numpy style <https://numpydoc.readthedocs.io/en/latest/format.html#id4>`_
* Pep8
* Use the following namespaces consistently:
.. code-block:: python
| import simtools.config as cfg
| import simtools.io_handler as io

DESIGN GUIDELINES (old, to be rewritten):
---------------

* Names must be validated. (See util/names.py and its documentation)
* Every functional class contains a 'label'. The label can be passed forward from lower level to higher level classes. In particular, label from TelescopeModel can be used in higher level classes.
* Classes are not designed to be re-used, all parameters should be set when initializing and not changed afterwards. New parameters should mean new instance of the class.
* filesLocation
* A test flag (test=True/False) should exist always when possible. If True, it must provide a faster and simpler implementation.
* Spectially important for time comsuming simulations. The default must always be test=False.
* A force flag (force=True/False) should exist in any case in which files are created. If False, the existing files should not be overwritten. The default must always be force=False.
* A clean method must be available for any class that create files.