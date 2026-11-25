# Simulate Calibration Data

simtools supports calibration simulations needed to configure, validate, and monitor CTAO
instrument response models. This section summarizes the main calibration simulation use cases.
Detailed command-line usage is documented in the [Applications](applications.md) reference.

## Pedestal events

Pedestal simulations provide camera baseline and noise information. They are used to study dark
pedestals, NSB-dependent pedestal shifts, and readout behavior without air-shower input.

## Flashers

Flasher simulations model controlled light delivered to the camera. They are used for gain
calibration, timing alignment, and cross-checks of photo-detection and readout response.

### FlashCam DC flasher

The DC flat-fielding device provides continuous light. Its properties, including wavelength and
angular distribution, are defined by the selected simulation model. The simulation represents its
contribution as an additional per-pixel rate:

```text
registered_rate_i = registered_nsb_rate_i + R_dc * w_i
```

* ``R_dc``: model parameter ``flasher_dc_pixel_rate`` (mean registered rate in GHz).
* ``w_i``: model parameter ``flasher_dc_relative_pixel_map`` (per pixel, mean 1).

The DC term is divided by sim_telarray's NSB scaling, mirror and camera degradation, and
photo-electron collection efficiency so that ``R_dc * w_i`` is the registered rate. A constant
``nsb_offaxis`` and no NSB sky map are required.

Set the DC rate to zero for an "off" run. The difference between matching "on" and "off" runs
isolates the DC contribution, ``R_dc * w_i``.

Run matching DC on and off simulations as shown in
{doc}`applications/simtools-simulate-flasher`. Choose a model version and array layout whose
telescopes have the intended DC device and relative pixel map configured. To derive a map for a
model, see {doc}`applications/simtools-derive-flasher-dc-pixel-map`.

The relative map reflects the simulated source distribution, optical acceptance, photon detection
efficiency and, with obdeect, structure shadowing. It does not include source non-uniformity unless
that is represented in the model used to derive the map.

## Muons

Muon simulations provide optical throughput calibration inputs based on ring images. The resulting
datasets support mirror and camera efficiency studies in pipeline validation workflows.

## Illuminators

Illuminator simulations describe external calibrated light sources used to probe telescope optics
and camera response under controlled conditions.
