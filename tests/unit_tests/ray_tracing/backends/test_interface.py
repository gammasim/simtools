#!/usr/bin/python3

from inspect import signature

import pytest

from simtools.ray_tracing.backends import interface
from simtools.ray_tracing.backends.interface import StarImageRequest


def test_star_image_request_key():
    request = StarImageRequest(0.5, 1.0, 10.0, ((1, 2),), 1000, mirror_number=3)
    assert request.key == (0.5, 1.0, 3)
    # Parameter changes do not distinguish images.
    changed = StarImageRequest(0.5, 1.0, 10.0, ((1, 2),), 1000, 3, {"focal_length": 1.0})
    assert changed == request


def test_get_backend(run_context):
    from simtools.ray_tracing.backends.sim_telarray import SimtelBackend

    assert interface.get_backend("sim_telarray") is SimtelBackend
    assert interface.get_backend(context=run_context) is SimtelBackend


def test_get_backend_obdeect():
    pytest.importorskip("obdeect")
    from simtools.ray_tracing.backends.obdeect import ObdeectBackend

    assert interface.get_backend("obdeect") is ObdeectBackend


@pytest.mark.parametrize(
    ("backend_name", "class_name", "expected_methods"),
    [
        (
            "sim_telarray",
            "SimtelBackend",
            {
                "simulate": (
                    "self",
                    "requests",
                    "test",
                    "force",
                    "compress_photons",
                    "max_workers",
                ),
                "psf_image": (
                    "self",
                    "request",
                    "focal_length",
                    "containment_fraction",
                    "use_rx",
                    "save_photons",
                ),
                "incidence_angles": (
                    "self",
                    "request",
                    "work_directory",
                    "name",
                    "with_mirrors",
                    "keep_files",
                ),
            },
        ),
        (
            "obdeect",
            "ObdeectBackend",
            {
                "simulate": (
                    "self",
                    "requests",
                    "test",
                    "force",
                    "compress_photons",
                    "max_workers",
                ),
                "psf_image": (
                    "self",
                    "request",
                    "focal_length",
                    "containment_fraction",
                    "use_rx",
                    "save_photons",
                ),
                "incidence_angles": (
                    "self",
                    "request",
                    "work_directory",
                    "name",
                    "with_mirrors",
                    "keep_files",
                ),
                "camera_photons": ("self", "request"),
            },
        ),
    ],
)
def test_backend_contract(backend_name, class_name, expected_methods):
    if backend_name == "obdeect":
        pytest.importorskip("obdeect")
    backend = interface.get_backend(backend_name)
    assert backend.__name__ == class_name
    for method_name, expected_parameters in expected_methods.items():
        assert tuple(signature(getattr(backend, method_name)).parameters) == expected_parameters


def test_get_backend_unknown_name():
    with pytest.raises(ValueError, match="sim_telarray, obdeect"):
        interface.get_backend("unknown")
