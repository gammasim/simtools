.. _Conventions:

Conventions
***********

.. _Telescope Names:

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




