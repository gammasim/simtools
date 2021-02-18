gammasim-tools
===========


Prototype implementation of tools for the Simulation System of the `CTA Observatory <www.cta-observatory.org>`_

gammasim-tools provides a framework and tools for:

* simulation model DB interface and management
* simulation model parameter derivation and validation

gammasim-tools follows these design principles:

* standardized interface and data products independent of the underlying software (e.g., CORSIKA, sim_telarray, GrOPTICS)
* maximizes the sharing of tools and algorithms with other DPPS subsystems (e.g., `ctapipe <https://github.com/cta-observatory/ctapipe>`_ and calibpipe)
* I/O and reporting tools for the MC simulation model parameter and production database

AUTHORS:
--------
    
* Raul R Prado (raul.prado@desy.de)
* Orel Gueta (orel.gueta@desy.de)
* Gernot Maier (gernot.maier@desy.de)

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

