#!/usr/bin/python3

import astropy.units as u

import simtools.config as cfg
from simtools.ray_tracing import RayTracing
from simtools.model.telescope_model import TelescopeModel


def test_config_data_from_dict(set_db):

    label = "test-config-data"
    version = "prod4"

    configData = {
        "sourceDistance": 10 * u.km,
        "zenithAngle": 30 * u.deg,
        "offAxisAngle": [0, 2] * u.deg,
    }

    tel = TelescopeModel(
        site="north",
        telescopeModelName="mst-FlashCam-D",
        modelVersion=version,
        label=label,
    )

    ray = RayTracing(telescopeModel=tel, configData=configData)

    assert ray.config.zenithAngle == 30
    assert len(ray.config.offAxisAngle) == 2


def test_from_kwargs(cfg_setup):

    label = "test-from-kwargs"

    sourceDistance = 10 * u.km
    zenithAngle = 30 * u.deg
    offAxisAngle = [0, 2] * u.deg

    cfgFile = cfg.findFile(
        "CTA-North-LST-1-Current_test-telescope-model.cfg",
        cfg.get("modelFilesLocations")
    )

    tel = TelescopeModel.fromConfigFile(
        site="north",
        telescopeModelName="lst-1",
        configFileName=cfgFile,
        label=label,
    )

    ray = RayTracing.fromKwargs(
        telescopeModel=tel,
        sourceDistance=sourceDistance,
        zenithAngle=zenithAngle,
        offAxisAngle=offAxisAngle,
    )

    assert ray.config.zenithAngle == 30
    assert len(ray.config.offAxisAngle) == 2
