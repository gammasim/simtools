.. _mcmodel:

MC Model
========

The gammasim-tools description of the MC Model has the `telescope_model`_
module (and its main class TelescopeModel) as its central element.
The TelescopeModel class provides a representation of a telescope in terms
of the MC Model. A TelescopeModel is defined by its telescope name.
An interface with the :ref:`Model Parameters DB`
is provided so the model parameters can be read and exported as configuration
files.

Lower level modules are defined to describe sub-elements of the TelescopeModel.
These modules are `mirrors`_ and `camera`_.

Arrays of telescopes are described by the Array Model module which contains a collection of TelescopeModel's and the array layout.

array_model
-----------

.. _array_model:

.. automodule:: model.array_model
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


model utitilities
-----------------

.. _model_utils:

.. automodule:: model.model_utils
   :members:

telescope_model
---------------

.. _telescope_model:

.. automodule:: model.telescope_model
   :members:
