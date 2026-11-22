"""Stream input and output of simulation stages."""

from dataclasses import dataclass

from simtools.simulation_software.programs import STDIO

#: Stream marker of a stage input read from the standard input of the chain.
STDIN = STDIO


@dataclass(frozen=True)
class StreamTo:
    """Output of a stage that is streamed to downstream stage commands (shell lines)."""

    commands: tuple[str, ...]
