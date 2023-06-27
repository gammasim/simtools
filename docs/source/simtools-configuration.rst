.. _Configuration:

Configuration
-------------

Applications in simtools can be configured by the following four approaches, which are all equivalent:

#. command line arguments;
#. configuration file (in yaml format);
#. configuration dictionary when calling the :ref:`Configurator <configurationconfigurator>` class;
#. environment variables.

To illustrate this, e.g., set the path pointing towards the directory for all data products.

Set the output directory using a command line argument:

.. code-block::

   $ python applications/<application_name> --output_path <path name>

Set the output directory using a configuration file in yaml format:

.. code-block::

   config_file: <path name>

Load the yaml configuration file into the application with:

.. code-block:: console

   $ python applications/<application_name> --config <my_config.yml>

Configuration parameter read from a environmental variable:

.. code-block:: console

   $ EXPORT OUTPUT_PATH="<path name>"

Configuration methods can be combined; conflicting configuration settings raise an Exception.
Configuration parameters are generally expected in lower-case snake-make case.
Configuration parameters for each application are printed to screen when executing the application with the ``--help`` option.
Parameters with the same functionality are named consistently the same among all applications.
