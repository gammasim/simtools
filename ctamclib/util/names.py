#!/usr/bin/python3

import logging


__all__ = ['validateName', 'isValidName', 'rayTracingFileName']


logger = logging.getLogger(__name__)


def validateName(name, allNames):
    if not isValidName(name, allNames):
        logger.error('Invalid name {}'.format(name))
        raise ValueError('Invalid name {}'.format(name))
    for mainName in allNames.keys():
        if name.lower() in allNames[mainName] + [mainName.lower()]:
            logger.debug('Correcting name {} -> {}'.format(name, mainName))
            return mainName


def isValidName(name, allNames):
    if not isinstance(name, str):
        return False
    for mainName in allNames.keys():
        if name.lower() in allNames[mainName] + [mainName.lower()]:
            return True
    return False


allTelescopeTypeNames = {
    'SST-1M': ['1m'],
    'SST-2M-ASTRI': ['sst-astri', 'astri'],
    'SST-2M-GCT-S': ['sst-gct', 'gct', 'sst-gct-s'],
    'MST-FlashCam': ['flashcam', 'mst-fc'],
    'MST-NectarCam': ['nectarcam', 'mst-nc'],
    'SCT': ['mst-sct', 'sct'],
    'LST': []
}

allSiteNames = {
    'Paranal': ['south'],
    'LaPalma': ['north']
}

allModelVersionNames = {
    'prod4': ['p4'],
    'default': []
}

allSimtelModeNames = {
    'RayTracing': ['raytracing', 'ray-tracing'],
    'Trigger': []
}


def rayTracingFileName(telescopeType, sourceDistance, zenithAngle, offAxisAngle, label, base):
    ''' base has to be log, stars or photons'''
    name = '{}-{}-d{:.1f}-za{:.1f}-off{:.3f}'.format(
        base,
        telescopeType,
        sourceDistance,
        zenithAngle,
        offAxisAngle
    )
    name += '_{}'.format(label) if label is not None else ''
    name += '.log' if base == 'log' else '.lis'
    return name


def rayTracingResultsFileName(telescopeType, sourceDistance, zenithAngle, label):
    name = 'ray-tracing-{}-d{:.1f}-za{:.1f}'.format(telescopeType, sourceDistance, zenithAngle)
    name += '_{}'.format(label) if label is not None else ''
    name += '.cvs'
    return name
