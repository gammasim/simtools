"""Registry for execution backend implementations."""

from simtools.job_execution.backends.base import BackendConfigurationError


def _local_factory():
    # Keep the local backend lazy so optional backends do not affect local use.
    # pylint: disable=import-outside-toplevel
    from simtools.job_execution.backends.local import (
        LocalBackend,
    )

    return LocalBackend()


def _htcondor_factory():
    # Keep htcondor2 lazy because it is an optional dependency.
    # pylint: disable=import-outside-toplevel
    from simtools.job_execution.backends.htcondor import (
        HTCondorBackend,
    )

    return HTCondorBackend()


_BACKENDS = {"local": _local_factory, "htcondor": _htcondor_factory}


def register_backend(name, factory):
    """Register a backend factory under ``name``."""
    if not name or not callable(factory):
        raise ValueError("Backend name and callable factory are required.")
    _BACKENDS[name] = factory


def available_backends():
    """Return registered backend names."""
    return tuple(sorted(_BACKENDS))


def get_backend(name):
    """Instantiate a registered backend."""
    try:
        return _BACKENDS[name]()
    except KeyError as exc:
        available = ", ".join(available_backends())
        raise BackendConfigurationError(
            f"Unknown execution backend '{name}'. Available backends: {available}."
        ) from exc
