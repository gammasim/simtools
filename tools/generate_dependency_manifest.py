"""Generate the installed simtools dependency manifest for a container image."""

import argparse
from pathlib import Path

from simtools.dependencies import write_dependency_manifest


def main():
    """Write the dependency manifest requested on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/opt/simtools/provenance/dependency-manifest.json"),
    )
    args = parser.parse_args()
    write_dependency_manifest(args.output)


if __name__ == "__main__":
    main()
