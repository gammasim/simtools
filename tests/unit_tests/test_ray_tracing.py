#!/usr/bin/python3

import astropy.units as u

import simtools.utils.general as gen
from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing import RayTracing


def test_config_data_from_dict(db_config, simtel_path, io_handler):
    label = "test-config-data"
    version = "prod5"

    config_data = {
        "source_distance": 10 * u.km,
        "zenith_angle": 30 * u.deg,
        "off_axis_angle": [0, 2] * u.deg,
    }

    tel = TelescopeModel(
        site="north",
        telescope_model_name="mst-FlashCam-D",
        model_version=version,
        label=label,
        mongo_db_config=db_config,
    )

    ray = RayTracing(
        telescope_model=tel,
        simtel_source_path=simtel_path,
        config_data=config_data,
    )

    assert ray.config.zenith_angle == 30
    assert len(ray.config.off_axis_angle) == 2


def test_from_kwargs(db, io_handler, simtel_path):
    label = "test-from-kwargs"

    source_distance = 10 * u.km
    zenith_angle = 30 * u.deg
    off_axis_angle = [0, 2] * u.deg

    test_file_name = "CTA-North-LST-1-Current_test-telescope-model.cfg"
    db.export_file_db(
        db_name="test-data",
        dest=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
        file_name=test_file_name,
    )

    cfg_file = gen.find_file(
        test_file_name, io_handler.get_output_directory(sub_dir="model", dir_type="test")
    )

    tel = TelescopeModel.from_config_file(
        site="north",
        telescope_model_name="lst-1",
        config_file_name=cfg_file,
        label=label,
    )

    ray = RayTracing.from_kwargs(
        telescope_model=tel,
        simtel_source_path=simtel_path,
        source_distance=source_distance,
        zenith_angle=zenith_angle,
        off_axis_angle=off_axis_angle,
    )

    assert ray.config.zenith_angle == 30
    assert len(ray.config.off_axis_angle) == 2
