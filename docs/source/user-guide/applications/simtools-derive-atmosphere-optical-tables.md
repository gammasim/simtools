# simtools-derive-atmosphere-optical-tables

Derive the atmospheric optical tables used by obdeect. The application computes the
optical properties on a wavelength and altitude grid with one backend and writes:

- a transmission table (`--transmission_file`), referenced to `--observation_level`,
- a scattering table (`--scattering_file`),
- optionally, the optical properties as an ECSV exchange file (`--properties_file`).

Backends:

- `6s`: runs the 6S radiative-transfer code. Requires Py6S and the 6S executable, found on
  `PATH` or given by `--sixs_path`. Wavelengths must be 250 to 4000 nm.
- `ecsv`: reads optical properties from an ECSV file (`--ecsv_file`) and resamples them to
  the requested grid.

Options that do not apply to the selected backend are rejected. Options left unset use the
backend defaults. Altitudes are given either as a list (`--altitudes`) or as a log-spaced
range (`--altitude_min`, `--altitude_max`, `--altitude_steps`); both forms are in m a.s.l.
and must be positive. Give sea level as a small positive altitude, for example `1`, because
the tables interpolate in log(altitude).

`--depolarization` sets the Rayleigh depolarization ratio in the scattering table (default 0,
no depolarization line).

The log reports the vertical transmission from the top of the altitude grid to
`--observation_level` at the shortest wavelength (350 nm if inside the grid) and at the longest
wavelength.

```{eval-rst}
.. automodule:: simtools.applications.derive_atmosphere_optical_tables
   :members:
   :exclude-members: main
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: derive_atmosphere_optical_tables
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
   :file: derive_atmosphere_optical_tables_ecsv.yml
```
