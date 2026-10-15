# simtools-derive-telescope-sphere-radius

Derive `telescope_sphere_radius` from the primary reflector and, where present,
the secondary reflector. The command writes `telescope_sphere_radius.json` in
metres and a companion `.meta.yml` file. The CORSIKA entrance sphere encloses
the mirror surfaces, including panel corners and dish curvature, for zenith
angles 0-90 degrees with no added margin. The camera is downstream of the
reflectors and is outside this entrance sphere. Missing or invalid mirror
geometry stops the derivation.

```{eval-rst}
.. automodule:: simtools.applications.derive_telescope_sphere_radius
   :members:
   :exclude-members: main
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: derive_telescope_sphere_radius
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
   :file: derive_telescope_sphere_radius_lstn.yml
```
