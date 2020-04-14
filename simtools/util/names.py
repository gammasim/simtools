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
            if name != mainName:
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
    'prod3_compatible': ['p3', 'prod3', 'prod3b'],
    'prod4': ['p4'],
    'default': []
}

allSimtelModeNames = {
    'RayTracing': ['raytracing', 'ray-tracing'],
    'RayTracingSingleMirror': [
        'raytracing-singlemirror',
        'ray-tracing-singlemirror',
        'ray-tracing-single-mirror'
    ],
    'Trigger': ['trigger']
}

allArrayNames = {
    '4LST': ['4-lst', '4lst'],
    '1LST': ['1-lst', '1lst'],
    '4MST': ['4-mst', '4mst'],
    '1MST': ['1-mst', 'mst'],
    '4SST': ['4-sst', '4sst'],
    '1SST': ['1-sst', 'sst']
}


def rayTracingFileName(
    telescopeType,
    sourceDistance,
    zenithAngle,
    offAxisAngle,
    mirrorNumber,
    label,
    base
):
    ''' base has to be log, stars or photons'''
    name = '{}-{}-d{:.1f}-za{:.1f}-off{:.3f}'.format(
        base,
        telescopeType,
        sourceDistance,
        zenithAngle,
        offAxisAngle
    )
    name += '_mirror{}'.format(mirrorNumber) if mirrorNumber is not None else ''
    name += '_{}'.format(label) if label is not None else ''
    name += '.log' if base == 'log' else '.lis'
    return name


def rayTracingResultsFileName(telescopeType, sourceDistance, zenithAngle, label):
    name = 'ray-tracing-{}-d{:.1f}-za{:.1f}'.format(telescopeType, sourceDistance, zenithAngle)
    name += '_{}'.format(label) if label is not None else ''
    name += '.cvs'
    return name


def corsikaConfigFileName(arrayName, site, zenith, viewCone, label=None):
    def isDifuse(viewCone):
        return viewCone[0] != 0 or viewCone[1] != 0

    name = 'corsika-config-{}-{}-za{:d}-{:d}'.format(arrayName, site, int(zenith[0]), int(zenith[1]))
    name += '-cone{:d}-{:d}'.format(int(viewCone[0]), int(viewCone[1])) if isDifuse(viewCone) else ''
    name += '_{}'.format(label) if label is not None else ''
    name += '.txt'
    return name


def corsikaOutputFileName(arrayName, site, zenith, viewCone, run, label=None):
    def isDifuse(viewCone):
        return viewCone[0] != 0 or viewCone[1] != 0

    name = 'corsika-run{}-{}-{}-za{:d}-{:d}'.format(
        run,
        arrayName,
        site,
        int(zenith[0]),
        int(zenith[1])
    )
    name += '-cone{:d}-{:d}'.format(int(viewCone[0]), int(viewCone[1])) if isDifuse(viewCone) else ''
    name += '_{}'.format(label) if label is not None else ''
    name += '.corsika.zst'
    return name
