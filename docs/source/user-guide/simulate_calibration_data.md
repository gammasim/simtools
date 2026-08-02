# Simulate Calibration Data

simtools supports calibration simulations needed to configure, validate, and monitor CTAO
instrument response models. This section summarizes the main calibration simulation use cases.
Detailed command-line usage is documented in the [Applications](applications.md) reference.

## Pedestal events

Pedestal simulations provide camera baseline and noise information. They are used to study dark
pedestals, NSB-dependent pedestal shifts, and readout behavior without air-shower input.

## Flashers

Flasher simulations model controlled light pulses injected into the camera. They are used for gain
calibration, timing alignment, and cross-checks of photo-detection and readout response.

## Muons

Muon simulations provide optical throughput calibration inputs based on ring images. The resulting
datasets support mirror and camera efficiency studies in pipeline validation workflows.

## Illuminators

Illuminator simulations describe external calibrated light sources used to probe telescope optics
and camera response under controlled conditions.
