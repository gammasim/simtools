"""
Simtools dependencies version management.

This modules provides two main functionalities:

- retrieve the versions of simtools dependencies (e.g., databases, sim_telarray, CORSIKA)
- provide space for future implementations of version management

"""

import hashlib
import json
import logging
import os
import platform
import re
import subprocess
from importlib import metadata
from pathlib import Path

import yaml

from simtools import settings
from simtools.io import ascii_handler
from simtools.utils import general as gen
from simtools.version import __version__

_logger = logging.getLogger(__name__)

DEPENDENCY_MANIFEST_PATH = Path("/opt/simtools/provenance/dependency-manifest.json")
DEPENDENCY_MANIFEST_SCHEMA_VERSION = "0.1.0"
SIMTEL_METADATA_BUILD_OPTION_KEYS = {
    "avx_flag",
    "build_date",
    "corsika_config_version",
    "corsika_opt_patch_version",
    "corsika_version",
    "extra_defines",
    "hessio_version",
    "iact_atmo_version",
    "simtel_version",
    "stdtools_version",
}


def get_version_string(run_time=None, include_software_versions=True):
    """
    Print the versions of the dependencies.

    Parameters
    ----------
    run_time : list, optional
        Runtime environment command (e.g., Docker).
    include_software_versions : bool, optional
        If True, query sim_telarray/CORSIKA executables and include their runtime versions.
        If False, skip executable checks and report these values as ``None``.

    Returns
    -------
    str
        String containing the versions of the dependencies.

    """
    simtel_version = None
    corsika_version = None
    simtel_exe = None
    corsika_exe = None

    def _safe_get_executable(path_getter):
        """Return executable path string, or None if environment is not configured."""
        try:
            return path_getter()
        except FileNotFoundError, TypeError:
            return None

    if include_software_versions:
        simtel_version = get_sim_telarray_version(run_time)
        corsika_version = get_corsika_version(run_time)
        simtel_exe = _safe_get_executable(lambda: settings.config.sim_telarray_exe)
        corsika_exe = _safe_get_executable(lambda: settings.config.corsika_exe)

    build_options = None
    if include_software_versions:
        build_options = get_build_options(run_time)

    return (
        f"simtools version: {__version__}\n"
        f"Database name: {get_database_version_or_name(version=False)}\n"
        f"Database version: {get_database_version_or_name(version=True)}\n"
        f"sim_telarray version: {simtel_version}\n"
        f"sim_telarray exe: {simtel_exe if simtel_exe else 'None'}\n"
        f"CORSIKA version: {corsika_version}\n"
        f"CORSIKA exe: {corsika_exe if corsika_exe else 'None'}\n"
        f"Build options: {build_options}\n"
        f"Runtime environment: {run_time if run_time else 'None'}\n"
    )


def get_dependency_manifest(run_time=None):
    """Return the installed dependency manifest or a discovered fallback.

    Parameters
    ----------
    run_time : list, optional
        Runtime command used to read a manifest from a container.

    Returns
    -------
    dict
        Dependency manifest.

    Raises
    ------
    FileNotFoundError
        If an explicitly requested container does not contain a manifest.
    ValueError
        If the manifest contains invalid JSON.
    """
    manifest_path = Path(os.getenv("SIMTOOLS_DEPENDENCY_MANIFEST", str(DEPENDENCY_MANIFEST_PATH)))
    try:
        return _read_dependency_manifest(manifest_path, run_time)
    except FileNotFoundError:
        if run_time is not None:
            raise
    return build_dependency_manifest()


def _read_dependency_manifest(manifest_path, run_time=None):
    """Read a dependency manifest locally or through a runtime command."""
    if run_time is None:
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise FileNotFoundError(f"Dependency manifest not found: {manifest_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid dependency manifest: {manifest_path}") from exc

    result = subprocess.run(
        [*run_time, "cat", str(manifest_path)], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise FileNotFoundError(f"Dependency manifest not found in container: {result.stderr}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid dependency manifest in container.") from exc


def build_dependency_manifest():
    """Build a deterministic manifest from the active runtime environment.

    Returns
    -------
    dict
        Discovered dependency manifest.
    """
    try:
        build_options = get_build_options()
    except FileNotFoundError, TypeError, ValueError:
        build_options = {}
    return {
        "schema_version": DEPENDENCY_MANIFEST_SCHEMA_VERSION,
        "source": (
            "container-build"
            if os.getenv("SIMTOOLS_CONTAINER_BUILD") == "1"
            else "runtime-discovery"
        ),
        "simtools": {
            "version": __version__,
            "revision": os.getenv("SIMTOOLS_GIT_REVISION") or _get_simtools_revision(),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "pip_version": _distribution_version("pip"),
            "direct_python_dependencies": get_direct_python_dependency_versions(),
        },
        "build_options": _sanitize_build_options(build_options),
        "container": {
            key: value
            for key, value in {
                "base_image": os.getenv("SIMTOOLS_BASE_IMAGE"),
                "corsika_image": os.getenv("SIMTOOLS_CORSIKA_IMAGE"),
                "sim_telarray_image": os.getenv("SIMTOOLS_SIMTEL_IMAGE"),
            }.items()
            if value
        },
    }


def _sanitize_build_options(build_options):
    """Remove nondeterministic values from build options used in a manifest."""
    return {key: value for key, value in build_options.items() if key != "build_date"}


def get_direct_python_dependency_versions():
    """Return installed versions of direct simtools Python dependencies.

    Returns
    -------
    dict
        Normalized distribution names mapped to installed versions.
    """
    try:
        requirements = metadata.requires("gammasimtools") or []
    except metadata.PackageNotFoundError:
        return {}
    versions = {}
    for requirement in requirements:
        if "extra ==" in requirement:
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)", requirement)
        if not match:
            continue
        name = match.group(1).lower().replace("_", "-")
        installed_version = _distribution_version(name)
        if installed_version is not None:
            versions[name] = installed_version
    return dict(sorted(versions.items()))


def _distribution_version(name):
    """Return an installed distribution version or None."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _get_simtools_revision():
    """Return the installed simtools VCS revision when available."""
    try:
        direct_url = metadata.distribution("gammasimtools").read_text("direct_url.json")
        vcs_info = json.loads(direct_url or "{}").get("vcs_info", {})
        return vcs_info.get("commit_id")
    except metadata.PackageNotFoundError, json.JSONDecodeError:
        return None


def canonical_manifest_bytes(manifest):
    """Serialize a manifest deterministically for hashing.

    Parameters
    ----------
    manifest : dict
        Dependency manifest.

    Returns
    -------
    bytes
        Canonical UTF-8 JSON representation.
    """
    return json.dumps(
        manifest,
        default=str,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def get_dependency_manifest_digest(run_time=None):
    """Return the SHA-256 digest of the dependency manifest."""
    manifest = get_dependency_manifest(run_time)
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def write_dependency_manifest(output_file):
    """Write the active dependency manifest and its SHA-256 file.

    Parameters
    ----------
    output_file : str or Path
        JSON manifest output path.
    """
    output_path = Path(output_file)
    manifest = build_dependency_manifest()
    content = json.dumps(manifest, default=str, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{digest}  {output_path.name}\n", encoding="utf-8"
    )


def get_dependency_summary(run_time=None):
    """Return a compact human-readable dependency summary."""
    manifest = get_dependency_manifest(run_time)
    build_options = manifest.get("build_options", {})
    simtools_info = manifest.get("simtools", {})
    runtime = manifest.get("runtime", {})
    return (
        f"simtools: {simtools_info.get('version', __version__)}\n"
        f"revision: {simtools_info.get('revision')}\n"
        f"Python: {runtime.get('python_version')}\n"
        f"CORSIKA: {build_options.get('corsika_version')}\n"
        f"sim_telarray: {build_options.get('simtel_version')}\n"
        f"dependency manifest SHA-256: "
        f"{hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()}"
    )


def get_dependency_metadata():
    """Return compact scalar provenance values for simulation metadata."""
    manifest = get_dependency_manifest()
    values = {
        "simtools_dependency_manifest_sha256": hashlib.sha256(
            canonical_manifest_bytes(manifest)
        ).hexdigest(),
        "simtools_git_revision": manifest.get("simtools", {}).get("revision"),
        "simtools_python_version": manifest.get("runtime", {}).get("python_version"),
    }
    return {key: value for key, value in values.items() if value is not None}


def get_software_version(software):
    """
    Return the version of the specified software package.

    Parameters
    ----------
    software : str
        Name of the software package.

    Returns
    -------
    str
        Version of the specified software package.
    """
    if software.lower() == "simtools":
        return __version__

    try:
        version_call = f"get_{software.lower()}_version"
        return globals()[version_call]()
    except KeyError as exc:
        raise ValueError(f"Unknown software: {software}") from exc


def get_database_version_or_name(version=True):
    """
    Get the version or name of the simulation model data base used.

    Parameters
    ----------
    version : bool
        If True, return the version of the database. If False, return the name.

    Returns
    -------
    str
        Version or name of the simulation model data base used.

    """
    if version:
        return settings.config.db_config and settings.config.db_config.get(
            "db_simulation_model_version"
        )
    return settings.config.db_config and settings.config.db_config.get("db_simulation_model")


def get_sim_telarray_version(run_time=None):
    """
    Get the version of the sim_telarray package using 'sim_telarray --version'.

    Version strings for sim_telarray are of the form "2024.271.0" (year.day_of_year.patch).

    Parameters
    ----------
    run_time : list, optional
        Runtime environment command (e.g., Docker).

    Returns
    -------
    str
        Version of the sim_telarray package.
    """
    if settings.config.sim_telarray_exe is None:
        _logger.warning("sim_telarray environment not configured.")
        return None
    if run_time is None:
        command = [str(settings.config.sim_telarray_exe), "--version"]
    else:
        command = [*run_time, str(settings.config.sim_telarray_exe), "--version"]

    _logger.debug(f"Running command: {command}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    # expect stdout with e.g. a line 'Release: 2024.271.0 from 2024-09-27'
    match = re.search(r"^Release:\s+(.+)", result.stdout, re.MULTILINE)
    if match:
        return match.group(1).split()[0]

    _logger.debug(f"Command output stdout: {result.stdout} stderr: {result.stderr}")

    raise ValueError(f"sim_telarray release not found in {result.stdout}")


def get_corsika_version(run_time=None):
    """
    Get the version of the CORSIKA package.

    Version strings for CORSIKA are of the form "7.7550" (major.minor with 4-digit minor).

    Parameters
    ----------
    run_time : list, optional
        Runtime environment command (e.g., Docker).

    Returns
    -------
    str
        Version of the CORSIKA package.
    """
    if settings.config.corsika_exe is None:
        _logger.warning("CORSIKA environment not configured.")
        return None

    if run_time is None:
        command = [str(settings.config.corsika_exe)]
    else:
        command = [*run_time, str(settings.config.corsika_exe)]

    process = subprocess.Popen(  # pylint: disable=consider-using-with
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True,
    )

    version = None
    for line in process.stdout:
        # Extract the version from the line "NUMBER OF VERSION :  7.7550"
        if "NUMBER OF VERSION" in line:
            version = line.split(":")[1].strip()
            break
        # Check for a specific prompt or indication that the program is waiting for input
        if "DATA CARDS FOR RUN STEERING ARE EXPECTED FROM STANDARD INPUT" in line:
            break

    process.terminate()
    # Check it's a valid version string
    if version and re.match(r"\d+\.\d+", version):
        return version
    try:
        build_opts = get_build_options(run_time)
    except FileNotFoundError, TypeError, ValueError:
        _logger.warning("Could not get CORSIKA version.")
        return None
    _logger.debug("Getting the CORSIKA version from the build options.")
    return build_opts.get("corsika_version")


def get_build_options(run_time=None):
    """
    Return CORSIKA / sim_telarray config and build options.

    For CORSIKA / sim_telarray build for simtools version >0.25.0:
    expects build_opts.yml file in each CORSIKA and sim_telarray
    directories.

    For CORSIKA / sim_telarray build for simtools version <=0.25.0:
    expects a build_opts.yml file in the sim_telarray directory.

    Parameters
    ----------
    run_time : list, optional
        Runtime environment command (e.g., Docker).

    Returns
    -------
    dict
        CORSIKA / sim_telarray build options.
    """
    build_opts = {}
    for package in ["corsika", "sim_telarray"]:
        path = _get_package_path(package)
        if not path:
            continue
        try:
            build_opts.update(_get_build_options_from_file(path / "build_opts.yml", run_time))
        except FileNotFoundError, TypeError, ValueError:
            # legacy fallback only for sim_telarray
            if package == "sim_telarray":
                try:
                    legacy_path = path.parent / "build_opts.yml"
                    build_opts.update(_get_build_options_from_file(legacy_path, run_time))
                except FileNotFoundError, TypeError, ValueError:
                    _logger.debug(f"No build options found for {package}.")
    if not build_opts:
        raise FileNotFoundError("No build option file found.")

    return build_opts


def _get_package_path(package):
    """Get the package path from settings or environment variables."""
    path = getattr(settings.config, f"{package}_path")
    if path is None:
        path = gen.load_environment_variables().get(f"{package}_path")
    return Path(path) if path else None


def _get_build_options_from_file(build_opts_path, run_time=None):
    """Read build options from file."""
    if run_time is None:
        try:
            return ascii_handler.collect_data_from_file(build_opts_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError("No build option file found.") from exc

    command = [*run_time, "cat", str(build_opts_path)]
    _logger.debug(f"Reading build option with command: {command}")

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise FileNotFoundError(f"No build option file found in container: {result.stderr}")

    try:
        return yaml.safe_load(result.stdout)
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing build_opts.yml from container: {exc}") from exc


def export_build_info(output_file, run_time=None):
    """
    Export build and version information to a file.

    Parameters
    ----------
    output_file : str
        Path to the output file.
    run_time : list, optional
        Runtime environment command (e.g., Docker).
    """
    try:
        build_options = get_build_options(run_time)
    except FileNotFoundError:
        build_options = {}
    manifest = get_dependency_manifest(run_time)
    database_name = get_database_version_or_name(version=False)
    database_version = get_database_version_or_name(version=True)
    build_info = {
        "schema_version": DEPENDENCY_MANIFEST_SCHEMA_VERSION,
        "dependency_manifest": manifest,
        "dependency_manifest_sha256": hashlib.sha256(
            canonical_manifest_bytes(manifest)
        ).hexdigest(),
        "runtime": {
            "database_name": database_name,
            "database_version": database_version,
        },
        "build_options": build_options,
        # Compatibility fields retained for existing consumers.
        **build_options,
        "simtools": __version__,
        "database_name": database_name,
        "database_version": database_version,
    }
    ascii_handler.write_data_to_file(data=build_info, output_file=Path(output_file))
