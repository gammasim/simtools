''' Parameters of telescope model. '''

__all__ = ['MODEL_PARS']

MODEL_PARS = {
    'focal_length': {'type': float, 'names': []},
    'fadc_mhz': {'type': float, 'names': ['fadc_MHz']},
    'quantum_efficiency': {'type': str, 'names': ['quantum_eff']},
    'telescope_transmission': {'type': str, 'names': ['telescope_trans']},
    'camera_transmission': {'type': str, 'names': ['camera_trans']}
}
