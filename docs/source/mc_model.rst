.. _mcmodel:

MC Model
========

In this section you find reference documentation for the modules
related to the MC Model. 

The simtools description of the MC Model has the :ref:`Telescope Model`
module (and its main class TelescopeModel) as its central element.
The TelescopeModel class provides a representation of a telescope in terms
of the MC Model. A TelescopeModel is basically defined by its telescope name
(see :ref:`Telescope Names`). An interface with the :ref:`Model Parameters DB`
is provided so the model parameters can be read and exported as configuration
files.

For the sake of modularity, lower level modules are defined to describe
sub-elements of the TelescopeModel. At the moment, these modules are
:ref:`Mirrors` and :ref:`Camera`.

Array of telescopes is described by the Array Model module (not implemented yet),
which contains a collection of TelescopeModel's and the array layout. 


* `Telescope Model`_
* `Mirrors`_
* `Camera`_


.. _Telescope Model:

Telescope Model
---------------

.. automodule:: model.telescope_model
   :members:


.. _Mirrors:

Mirrors
-------

.. automodule:: model.mirrors
   :members:

.. _Camera:


Camera
------

.. automodule:: model.camera
   :members:
