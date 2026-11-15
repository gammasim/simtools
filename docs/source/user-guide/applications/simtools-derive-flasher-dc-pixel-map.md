# simtools-derive-flasher-dc-pixel-map

```{eval-rst}
.. automodule:: simtools.applications.derive_flasher_dc_pixel_map
   :members:
   :exclude-members: main
```

```{eval-rst}
Derive the relative per-pixel illumination map of a DC flat-fielding device (e.g. ``MSFx-FlashCam-DC``)
from a low-intensity flasher simulation. The application sums the true photo-electrons per pixel
over all events and writes the map normalised to a mean of 1 (ECSV columns: ``pixel_id``,
``relative_rate``).

The map includes light-concentrator acceptance, photon detection efficiency and, with the obdeect
ray-tracing backend, telescope-structure shadowing. The emitter itself is an ideal Lambertian source.

Requirements for the input simulations:

* ``simtools-simulate-flasher --run_mode full_simulation`` with sim_telarray configured to store
  photo-electrons (``store_photoelectrons``).
* Night-sky background and camera noise switched off.
* Enough events to reach the required precision in every pixel; inspect the output uncertainties.
  Keep the event intensity below sim_telarray's photon-bunch limit.

**Example**

.. code-block:: console

    simtools-derive-flasher-dc-pixel-map \\
        --simtel_files "/path/to/dc_map/*.simtel.zst" \\
        --telescope MSTN-04 \\
        --output_file flasher_dc_relative_pixel_map.ecsv \\
        --plot --output_path dc_map

```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: derive_flasher_dc_pixel_map
   :no-heading:
```
