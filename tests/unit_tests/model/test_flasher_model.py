"""Unit tests for the FlasherModel class."""

import unittest.mock as mock

from simtools.model.flasher_model import FlasherModel


def test_flasher_model_init():
    """Test FlasherModel initialization."""
    with mock.patch("simtools.model.model_parameter.ModelParameter.__init__") as mock_init:
        mock_init.return_value = None
        FlasherModel(
            site="North",
            flasher_device_model_name="FLSN-01",
            mongo_db_config={},
            model_version="6.0.0",
            label="test_label",
        )

        mock_init.assert_called_once_with(
            site="North",
            array_element_name="FLSN-01",
            collection="flasher_devices",
            mongo_db_config={},
            model_version="6.0.0",
            db=None,
            label="test_label",
        )
