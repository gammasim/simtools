"""Content-derived identities for figures shared by documentation pages."""

import hashlib
import importlib.metadata
import json
import locale
from functools import lru_cache
from pathlib import Path

import numpy as np

from simtools.settings import config as simtools_config
from simtools.visualization import matplotlib_backend
from simtools.visualization.matplotlib_backend import BACKEND_NAME


def figure_cache_key(renderer, renderer_modules, plot_config, rendered_data):
    """Return a SHA-256 key for identical plot inputs and rendering settings."""
    renderer_modules = {
        "figure_cache": __file__,
        "matplotlib_backend": matplotlib_backend.__file__,
        **renderer_modules,
    }
    renderer_source = {
        name: _source_digest(Path(module_path).resolve())
        for name, module_path in renderer_modules.items()
    }
    output_settings = {
        key: simtools_config.args.get(key)
        for key in ("figure_format", "figure_dpi", "figure_png_palette")
    }
    output_settings.update(
        {
            "locale": locale.setlocale(locale.LC_ALL),
            "matplotlib_backend": BACKEND_NAME,
            "matplotlib_version": importlib.metadata.version("matplotlib"),
        }
    )
    payload = {
        "cache_format": 1,
        "renderer": renderer,
        "renderer_source": renderer_source,
        "plot_config": plot_config,
        "rendered_data": rendered_data,
        "output_settings": output_settings,
    }
    canonical_payload = json.dumps(
        _canonicalize(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


@lru_cache(maxsize=8)
def _source_digest(path):
    """Return a SHA-256 digest for a renderer source module."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonicalize(value):
    """Convert NumPy, dataclass and structured plot inputs into stable JSON values."""
    if hasattr(value, "__dataclass_fields__"):
        value = {field: getattr(value, field) for field in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.descr if value.dtype.names else value.dtype.str,
            "shape": list(value.shape),
            "values": _canonicalize(value.tolist()),
        }
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    return value
