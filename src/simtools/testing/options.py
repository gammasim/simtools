"""Helpers for pytest options mirrored by environment variables."""

import os

MIRRORED_OPTIONS = {
    "simulation_models_path": (
        "SIMTOOLS_SIMULATION_MODELS_PATH",
        "Read simulation models from files at this path.",
    ),
    "simulation_models_git_path": (
        "SIMTOOLS_SIMULATION_MODELS_GIT_PATH",
        "Read simulation models from a local Git repository at this path.",
    ),
    "simulation_models_git_revision": (
        "SIMTOOLS_SIMULATION_MODELS_GIT_REVISION",
        "Git revision used for the simulation-model reader.",
    ),
    "simtools_tests_path": (
        "SIMTOOLS_TESTS_PATH",
        "Root of the versioned simtools-tests repository.",
    ),
    "simtools_tests_resource_version": (
        "SIMTOOLS_TESTS_RESOURCE_VERSION",
        "Versioned simtools-tests resource directory.",
    ),
}


def add_mirrored_options(parser):
    """Add pytest options whose values can also come from environment variables."""
    for option_name, (_, help_text) in MIRRORED_OPTIONS.items():
        parser.addoption(
            f"--{option_name}",
            dest=option_name,
            default=None,
            help=help_text,
        )


def get_mirrored_option(config, option_name):
    """Return a command-line option, falling back to its environment variable."""
    value = config.getoption(option_name, default=None)
    if value is not None:
        return value
    return os.environ.get(MIRRORED_OPTIONS[option_name][0])
