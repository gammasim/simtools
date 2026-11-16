"""Stream input and output of simulation stages (the products are in ``products``)."""

from dataclasses import dataclass

STDIN = "-"


@dataclass(frozen=True)
class StreamTo:
    """Output of a stage that is streamed to downstream stage commands (shell lines)."""

    commands: tuple[str, ...]
