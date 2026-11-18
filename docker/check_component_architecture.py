"""Check the architecture of the simulation executables in a container image."""

import sys
from pathlib import Path

import yaml


def main():
    """Fail if a component executable does not match the target architecture."""
    target, simtel, corsika = sys.argv[1:]
    machines = {"amd64": 62, "arm64": 183}
    if target not in machines:
        raise SystemExit(f"Unsupported target architecture: {target}")
    corsika_directory = Path(corsika)
    manifest = corsika_directory / "build_opts.yml"
    with manifest.open(encoding="utf-8") as stream:
        options = yaml.safe_load(stream)
    if not isinstance(options, dict) or not isinstance(options.get("variant"), list):
        raise SystemExit(f"No CORSIKA variant list in {manifest}")
    variants = options["variant"]
    if not variants or any(
        not isinstance(entry, dict) or not entry.get("executable") for entry in variants
    ):
        raise SystemExit(f"No CORSIKA executables declared in {manifest}")
    corsika_executables = [corsika_directory / entry["executable"] for entry in variants]
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
