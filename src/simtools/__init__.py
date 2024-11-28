import logging

from .version import __version__

logging.basicConfig(format="%(levelname)s::%(module)s(l%(lineno)s)::%(funcName)s::%(message)s")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

__all__ = ["__version__"]
