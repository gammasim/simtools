"""Compatibility access to shared figure-cache identities."""

from simtools.visualization import visualize

BACKEND_NAME = visualize.BACKEND_NAME


def figure_cache_key(renderer, renderer_modules, plot_config, rendered_data, context=None):
    """Return a SHA-256 key for identical plot inputs and rendering settings.

    The reporting-module import path remains available for existing callers. The implementation
    is shared by visualization renderers so they use the same identity.
    """
    return visualize.figure_cache_key(
        renderer, renderer_modules, plot_config, rendered_data, context
    )
