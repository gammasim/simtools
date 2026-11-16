"""Shared data products and stream output for simulation stages."""

from dataclasses import dataclass

CHERENKOV_PHOTONS = "cherenkov_photons"
CAMERA_EVENTS = "camera_events"
STDIN = "-"


@dataclass(frozen=True)
class StreamTo:
    """Output of a stage that is streamed to downstream stage commands (shell lines)."""

    commands: tuple[str, ...]
