"""Names of the photon products that connect the stages of a simulation chain.

A product is named by the frame of its photons and its format. The obdeect stages exchange
photon streams (``stream`` component of obdeect); the IACT products are CORSIKA/sim_telarray
EventIO files or streams. A stage consumes one or several products and produces one.
"""

#: CORSIKA IACT photon bunches in the ground frame (EventIO); output of the CORSIKA stage.
CHERENKOV_PHOTONS = "cherenkov_photons"
#: Photon stream, photons emitted at a point in the sky (light sources).
PHOTONS_SKY = "photons_sky"
#: Photon stream, ground frame: horizontal plane through the telescope sphere centre.
PHOTONS_GROUND = "photons_ground"
#: Photon stream, telescope frame (sources inside the telescope, e.g. flashers).
PHOTONS_TELESCOPE = "photons_telescope"
#: Photon stream, camera frame: position on the camera entrance, direction and time.
PHOTONS_CAMERA = "photons_camera"
#: Camera-frame photons as IACT photon bunches (EventIO), for sim_telarray with bypassed optics.
CAMERA_PHOTONS_IACT = "camera_photons_iact"
#: sim_telarray output (camera events).
CAMERA_EVENTS = "camera_events"
