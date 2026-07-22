"""Generate dependency provenance for a simtools development container."""

import hashlib
import importlib.metadata
import json
import os
import re
import tomllib
from pathlib import Path

import yaml

OUTPUT_PATH = Path("/opt/simtools/provenance/dependency-manifest.json")


def _direct_dependencies():
    """Return installed versions of direct project dependencies."""
    with Path("/workdir/pyproject.toml").open("rb") as file:
        requirements = tomllib.load(file)["project"]["dependencies"]
    versions = {}
    for requirement in requirements:
        name = re.match(r"^([A-Za-z0-9_.-]+)", requirement).group(1)
        versions[name.lower().replace("_", "-")] = importlib.metadata.version(name)
    return dict(sorted(versions.items()))


def _build_options():
    """Merge available CORSIKA and sim_telarray build options."""
    options = {}
    for path in (
        Path("/workdir/simulation_software/corsika7/build_opts.yml"),
        Path("/workdir/simulation_software/sim_telarray/build_opts.yml"),
    ):
        if path.exists():
            options.update(
                {
                    key: value
                    for key, value in (
                        yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                    ).items()
                    if key != "build_date"
                }
            )
    return options


def _canonical_bytes(manifest):
    """Return deterministic JSON bytes."""
    return json.dumps(manifest, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def main():
    """Write the development-image dependency manifest."""
    manifest = {
        "schema_version": "0.1.0",
        "source": "container-build",
        "simtools": {
            "version": "not-installed",
            "revision": os.environ.get("SIMTOOLS_GIT_REVISION"),
        },
        "runtime": {
            "python_version": os.sys.version.split()[0],
            "pip_version": importlib.metadata.version("pip"),
            "direct_python_dependencies": _direct_dependencies(),
        },
        "build_options": _build_options(),
        "container": {
            "base_image": os.environ.get("SIMTOOLS_BASE_IMAGE"),
            "corsika_image": os.environ.get("SIMTOOLS_CORSIKA_IMAGE"),
            "sim_telarray_image": os.environ.get("SIMTOOLS_SIMTEL_IMAGE"),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    OUTPUT_PATH.with_suffix(".json.sha256").write_text(
        f"{digest}  {OUTPUT_PATH.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
