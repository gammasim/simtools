"""Optional native bindings for sim_telarray LightEmission.

This module provides a thin Python wrapper around a native extension that can
invoke the LightEmission library directly (ff-1m first; xyzls later).

If the extension isn't available, ``HAS_NATIVE`` is False and callers should
gracefully fall back to the subprocess-based execution.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # help static analyzers without requiring the compiled module at runtime
    NATIVE: Any

try:  # native module is optional
    NATIVE = importlib.import_module("simtools.light_emission._le")
    HAS_NATIVE = True
except ImportError:
    NATIVE = None  # type: ignore[assignment]
    HAS_NATIVE = False


def run_ff_1m_native(
    output_path: Path,
    altitude_m: float,
    atmosphere_id: int,
    photons: float,
    bunch_size: int,
    x_cm: float,
    y_cm: float,
    distance_cm: float,
    camera_radius_cm: float,
    spectrum_nm: int,
    lightpulse: str,
    angular_distribution: str,
) -> None:
    """Run ff-1m via native bindings and write an IACT file.

    Parameters are a minimal subset to mirror the CLI options we use today.
    """
    if not HAS_NATIVE or NATIVE is None:
        raise RuntimeError("Native LightEmission bindings not available")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _logger.debug(
        "Running native ff-1m to %s (photons=%s, bunch=%s, xy=(%s,%s), d=%s cm)",
        output_path,
        photons,
        bunch_size,
        x_cm,
        y_cm,
        distance_cm,
    )

    rc = NATIVE.ff_1m(
        str(output_path),
        float(altitude_m),
        int(atmosphere_id),
        float(photons),
        int(bunch_size),
        float(x_cm),
        float(y_cm),
        float(distance_cm),
        float(camera_radius_cm),
        int(spectrum_nm),
        str(lightpulse),
        str(angular_distribution),
    )
    if rc != 0:
        raise RuntimeError(f"ff-1m native call failed with code {rc}")
