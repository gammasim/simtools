"""Check the architecture of the simulation executables in a container image."""

import sys
from pathlib import Path


def main():
    """Fail if a component executable does not match the target architecture."""
    target, simtel, corsika = sys.argv[1:]
    machines = {"amd64": 62, "arm64": 183}
    if target not in machines:
        raise SystemExit(f"Unsupported target architecture: {target}")
    corsika_executables = sorted(Path(corsika).glob("corsika_*"))
    if not corsika_executables:
        raise SystemExit(f"No CORSIKA executables found in {corsika}")
    for executable in [Path(simtel), *corsika_executables]:
        with executable.open("rb") as stream:
            header = stream.read(20)
        if (
            len(header) != 20
            or header[:6] != b"\x7fELF\x02\x01"
            or int.from_bytes(header[18:20], "little") != machines[target]
        ):
            raise SystemExit(f"{executable} is not a {target} ELF executable")


if __name__ == "__main__":
    main()
