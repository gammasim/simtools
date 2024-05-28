.. _mcmodel:

MC Model
========

The array of imaging atmospheric Cherenkov telescopes is abstracted in the simulation model and divided into the following components:

- `telescope_model`_ representing a telescope. Defined by its telescope name, allowing to read model parameters from the databases using this name.
- sub-elements of the telescope represented by the modules `mirrors`_ and `camera`_
- an array of telescopes (especially the telescope arrangement) represented by `array_model`_.



array_model
-----------

.. _array_model:

.. automodule:: model.array_model
   :members:


calibration_model
-----------------

.. _calibration_model:

.. automodule:: model.calibration_model
   :members:

camera
------

.. _camera:

.. automodule:: model.camera
   :members:


mirrors
-------

.. _mirrors:

.. automodule:: model.mirrors
   :members:


model_parameter
---------------

.. _model_parameters:

.. automodule:: model.model_parameter
   :members:


model utilities
---------------

.. _model_utils:

.. automodule:: model.model_utils
   :members:

site_model
----------

.. _site_model:

.. automodule:: model.site_model
   :members:

telescope_model
---------------

.. _telescope_model:

.. automodule:: model.telescope_model
   :members:
