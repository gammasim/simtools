#!/usr/bin/python3

__all__ = ['MODEL_PARS', 'TWO_MIRROR_TELS', 'CAMERA_ROTATE_ANGLE']

MODEL_PARS = {
    'focal_length': {'type': float, 'names': []},
    'fadc_mhz': {'type': float, 'names': ['fadc_MHz']},
    'quantum_efficiency': {'type': str, 'names': ['quantum_eff']},
    'telescope_transmission': {'type': str, 'names': ['telescope_trans']},
    'camera_transmission': {'type': str, 'names': ['camera_trans']}
}

TWO_MIRROR_TELS = ['SCT', 'SST-2M-ASTRI', 'SST-2M-GCT-S', 'SST-Structure', 'SST-Camera']

# The coordinate system is aligned with Alt (x) and Az(y), so need to rotate the camera.
# The angle depends on what coordinate system was provided by the instrument team.
# Specifically the case of LST and NectarCam is a bit weird at 270 - 2*10.893,
# where the factor two is because we need to undo the rotation in the code
# and then actually rotate in the right direction.
CAMERA_ROTATE_ANGLE = {
  'LST': 248.214,
  'MST-FlashCam': 270,
  'MST-NectarCam': 248.214,
  'SCT': 90,
  'SST-2M-ASTRI': 90,
  'SST-1M': 270,
  'SST-2M-GCT-S': 90,
  'SST-Structure': 90,
  'SST-Camera': 90
}
