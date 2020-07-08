''' Parameters of telescope model. '''

__all__ = ['MODEL_PARS', 'TWO_MIRROR_TELS', 'CAMERA_ROTATE_ANGLE']

MODEL_PARS = {
    'focal_length': {'type': float, 'names': []},
    'fadc_mhz': {'type': float, 'names': ['fadc_MHz']},
    'quantum_efficiency': {'type': str, 'names': ['quantum_eff']},
    'telescope_transmission': {'type': str, 'names': ['telescope_trans']},
    'camera_transmission': {'type': str, 'names': ['camera_trans']},
    'trigger_telescopes': {'type': int, 'names': []},
    'iobuf_maximum': {'type': int, 'names': ['iobuf_max']},
    'iobuf_output_maximum': {'type': int, 'names': ['iobuf_output_max']},
    'only_triggered_telescopes': {'type': int, 'names': []},
    'store_photoelectrons': {'type': int, 'names': []},
    'sum_after_peak': {'type': int, 'names': []},
    'sum_before_peak': {'type': int, 'names': []},
    'trigger_pixels': {'type': int, 'names': ['trigger_pixel']},
    'pixel_cells': {'type': int, 'names': []},
    'pixels_parallel': {'type': int, 'names': []},
    # 'pm_transit_time': {'type': int, 'names': []},
    'pulse_analysis': {'type': int, 'names': []},
    'num_gains': {'type': int, 'names': []},
    'output_format': {'type': int, 'names': []},
    'peak_sensing': {'type': int, 'names': []},
    'fadc_sum_offset': {'type': int, 'names': []},
    'flatfielding': {'type': int, 'names': []},
    'mirror_class': {'type': int, 'names': []},
    'fadc_max_signal': {'type': int, 'names': []},
    'fadc_max_sum': {'type': int, 'names': []},
    'fadc_sum_bins': {'type': int, 'names': []},
    'fadc_bins': {'type': int, 'names': []},
    'fadc_lg_max_signal': {'type': int, 'names': []},
    'fadc_lg_max_sum': {'type': int, 'names': []},
    'disc_bins': {'type': int, 'names': []},
    'disc_start': {'type': int, 'names': []},
    'fadc_ac_coupled': {'type': int, 'names': []},
    'camera_pixels': {'type': int, 'names': ['camera_pixel']},
    'disc_ac_coupled': {'type': int, 'names': []},
    'camera_body_shape': {'type': int, 'names': []},
    'min_photoelectrons': {'type': int, 'names': []},
    'parabolic_dish': {'type': int, 'names': []}
}

TWO_MIRROR_TELS = ['SCT', 'SST-2M-ASTRI', 'SST-2M-GCT-S', 'SST-Structure', 'SST-Camera', 'SST']

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
  'SST-Camera': 90,
  'SST': 90
}

CAMERA_RADIUS_CURV = {
    'SST': 4.566,
    'SST-1M': 5.6,
    'SST-2M-ASTRI': 4.3,
    'SST-2M-GCT-S': 4.566,
    'MST-FlashCam': 19.2,
    'MST-NectarCam': 19.2,
    'MST-SCT': 11.16,
    'LST': 56
}
