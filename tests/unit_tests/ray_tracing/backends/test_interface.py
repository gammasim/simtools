#!/usr/bin/python3

import pytest

from simtools.ray_tracing.backends import interface
from simtools.ray_tracing.backends.interface import StarImageRequest


def test_star_image_request_key():
    request = StarImageRequest(0.5, 1.0, 10.0, ((1, 2),), 1000, mirror_number=3)
    assert request.key == (0.5, 1.0, 3)
    # Parameter changes do not distinguish images.
    changed = StarImageRequest(0.5, 1.0, 10.0, ((1, 2),), 1000, 3, {"focal_length": 1.0})
    assert changed == request


def test_get_backend(mocker):
    from simtools.ray_tracing.backends.sim_telarray import SimtelBackend

    assert interface.get_backend("sim_telarray") is SimtelBackend
    settings = mocker.patch.object(interface, "settings")
    settings.config.ray_tracing_backend = "sim_telarray"
    assert interface.get_backend() is SimtelBackend


def test_get_backend_obdeect():
    pytest.importorskip("obdeect")
    from simtools.ray_tracing.backends.obdeect import ObdeectBackend

    assert interface.get_backend("obdeect") is ObdeectBackend
