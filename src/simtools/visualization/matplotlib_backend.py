"""Configure the Matplotlib backend used by simtools visualizations."""

from importlib import import_module


class _MatplotlibModuleProxy:
    """Load a plotting dependency only when a visualization uses it."""

    def __init__(self, module_name):
        self._module_name = module_name
        self._module = None

    def _load(self):
        """Configure and return the requested Matplotlib module."""
        if self._module is None:
            matplotlib = import_module("matplotlib")
            matplotlib.use("Agg")
            self._module = import_module(self._module_name)
        return self._module

    def __getattr__(self, name):
        """Resolve module attributes after loading the plotting backend."""
        return getattr(self._load(), name)

    def __getitem__(self, key):
        """Resolve indexed module objects after loading the plotting backend."""
        return self._load()[key]


def lazy_module(module_name):
    """Return a proxy that imports a plotting dependency on first use.

    Parameters
    ----------
    module_name : str
        Fully qualified name of the plotting module to import lazily.

    Returns
    -------
    _MatplotlibModuleProxy
        Proxy that loads ``module_name`` when an attribute or item is accessed.
    """
    return _MatplotlibModuleProxy(module_name)


pyplot = lazy_module("matplotlib.pyplot")
