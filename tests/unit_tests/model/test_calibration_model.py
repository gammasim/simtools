#!/usr/bin/python3

import logging

from simtools.model.calibration_model import CalibrationModel


def test_calibration_model_init(caplog):
    with caplog.at_level(logging.DEBUG):
        model = CalibrationModel(
            site="North",
            calibration_device_model_name="ILLN-01",
            model_version="1.0.0",
        )

    assert model.site == "North"
    assert model.name == "ILLN-01"
