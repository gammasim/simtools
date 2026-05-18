"""
Thin adapter exposing the legacy production-grid API.

The backend-independent implementation lives in
``simtools.production_configuration.production_grid``.
"""

from simtools.production_configuration.production_grid_engine import ProductionGridEngine


class GridGeneration(ProductionGridEngine):
    # pylint: disable=too-few-public-methods
    """
    Backward-compatible adapter for production-grid generation.

    This class preserves the legacy public API while delegating the full
    implementation to
    :class:`~simtools.production_configuration.production_grid.ProductionGridEngine`.
    """
