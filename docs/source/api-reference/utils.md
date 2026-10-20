(utilities)=

# Utilities

In this section you find the reference documentation of the modules contained in
the util module.

## constants

```{eval-rst}
.. automodule:: constants
   :members:
```

(utilsgeneral)=

## general

```{eval-rst}
.. automodule:: utils.general
   :members:

```

(utilsgeometry)=

## geometry

`telescope_fiducial_sphere_radius` calculates a conservative CORSIKA sphere from
cylindrical bounds on optical elements. Include every relevant element and its full
surface sag and panel corners. Coordinates are relative to the mirror backplane;
`mirror_offset` and both `axes_offsets` use sim_telarray conventions. The default
pointing range is zenith angles from 0 to 90 degrees. Any extra clearance must be
specified explicitly.

For the MST reference enclosure:

```python
import astropy.units as u
from simtools.utils.geometry import telescope_fiducial_sphere_radius

radius = telescope_fiducial_sphere_radius(
    radial_extent=[6.90] * u.m,
    axial_extent=[[0, 1.28]] * u.m,
    axes_offsets=[1.60, 0] * u.m,
    mirror_offset=-1.775 * u.m,
)
# radius is approximately 9.15 m; additional elements require their own bounds.
```

```{eval-rst}
.. automodule:: utils.geometry
   :members:
```

(utilsnames)=

## names

```{eval-rst}
.. automodule:: utils.names
   :members:
```

## random numbers

```{eval-rst}
.. automodule:: utils.random
   :members:
```

## value_conversion

```{eval-rst}
.. automodule:: utils.value_conversion
   :members:
```
