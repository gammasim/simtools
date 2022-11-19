.. _mcmodel:

MC Model
========

The gammasim-tools description of the MC Model has the :ref:`telescope_model`
module (and its main class TelescopeModel) as its central element.
The TelescopeModel class provides a representation of a telescope in terms
of the MC Model. A TelescopeModel is defined by its telescope name.
An interface with the :ref:`Model Parameters DB`
is provided so the model parameters can be read and exported as configuration
files.

Lower level modules are defined to describe sub-elements of the TelescopeModel.
These modules are :ref:`mirrors` and :ref:`camera`.

Arrays of telescopes are described by the Array Model module which contains a collection of TelescopeModel's and the array layout.

array_model
-----------

.. automodule:: model.array_model
   :members:

.. _model_utils:


.. _telescope_model:

camera
------

.. automodule:: model.camera
   :members:

.. _array_model:

mirrors
-------

.. automodule:: model.mirrors
   :members:

.. _camera:



model utitilities
-----------------

.. automodule:: model.model_utils
   :members:

telescope_model
---------------

.. automodule:: model.telescope_model
   :members:


.. _mirrors:
