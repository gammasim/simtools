"""Read and validate CORSIKA build variants."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from simtools.configuration import defaults


@dataclass(frozen=True)
class CorsikaBuildVariant:
    """One executable variant declared by a CORSIKA build."""

    executable: str
    config: str
    atmosphere_geometry: str
    he_hadronic_model: str
    le_hadronic_model: str

    @classmethod
    def from_mapping(cls, entry):
        """Create a variant from a build-options mapping.

        Parameters
        ----------
        entry : dict
            One entry from the ``variant`` list in ``build_opts.yml``.

        Returns
        -------
        CorsikaBuildVariant
            Validated and normalized variant.

        Raises
        ------
        ValueError
            If a required field is missing or invalid.
        """
        if not isinstance(entry, dict):
            raise ValueError("Invalid CORSIKA build variant; expected a mapping")
        required = (
            "executable",
            "config",
            "atmosphere_geometry",
            "he_hadronic_model",
            "le_hadronic_model",
        )
        missing = [key for key in required if not entry.get(key)]
        if missing:
            raise ValueError(
                "Invalid CORSIKA build variant; missing field(s): " + ", ".join(missing)
            )

        geometry = str(entry["atmosphere_geometry"]).lower()
        if geometry not in {"flat", "curved"}:
            raise ValueError(f"Invalid CORSIKA atmosphere geometry: {geometry}")

        return cls(
            executable=str(entry["executable"]),
            config=str(entry["config"]),
            atmosphere_geometry=geometry,
            he_hadronic_model=str(entry["he_hadronic_model"]).lower(),
            le_hadronic_model=str(entry["le_hadronic_model"]).lower(),
        )


def read_corsika_build_variants(corsika_path):
    """Read CORSIKA executable variants from ``build_opts.yml``.

    Parameters
    ----------
    corsika_path : str or Path
        CORSIKA installation directory.

    Returns
    -------
    tuple of CorsikaBuildVariant
        Installed executable variants.

    Raises
    ------
    FileNotFoundError
        If the build-options file is unavailable.
    ValueError
        If the file is malformed, contains no variants, or declares duplicates.
    """
    build_options_path = Path(corsika_path) / "build_opts.yml"
    try:
        with open(build_options_path, encoding="utf-8") as build_options_file:
            build_options = yaml.safe_load(build_options_file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid CORSIKA build options in {build_options_path}: {exc}") from exc

    if not isinstance(build_options, dict) or not isinstance(build_options.get("variant"), list):
        raise ValueError(f"CORSIKA build options contain no variant list: {build_options_path}")

    variants = tuple(CorsikaBuildVariant.from_mapping(entry) for entry in build_options["variant"])
    if not variants:
        raise ValueError(
            f"CORSIKA build options contain an empty variant list: {build_options_path}"
        )

    variant_keys = [
        (variant.he_hadronic_model, variant.le_hadronic_model, variant.atmosphere_geometry)
        for variant in variants
    ]
    if len(variant_keys) != len(set(variant_keys)):
        raise ValueError(f"Duplicate CORSIKA build variants in {build_options_path}")
    return variants


def get_installed_corsika_build_variants(corsika_path):
    """Return build variants whose declared executables are installed.

    Parameters
    ----------
    corsika_path : str or Path
        CORSIKA installation directory.

    Returns
    -------
    tuple of CorsikaBuildVariant
        Validated installed variants.

    Raises
    ------
    FileNotFoundError
        If the manifest is missing.
    ValueError
        If the manifest declares a missing executable.
    PermissionError
        If a declared executable is not executable.
    """
    corsika_directory = Path(corsika_path)
    variants = read_corsika_build_variants(corsika_directory)
    for variant in variants:
        executable = corsika_directory / variant.executable
        if not executable.is_file():
            raise ValueError(f"CORSIKA build manifest declares a missing executable: {executable}")
        if not os.access(executable, os.X_OK):
            raise PermissionError(
                f"CORSIKA build manifest declares a non-executable file: {executable}"
            )
    return variants


def select_corsika_build_variant(variants, he_model, le_model, geometry):
    """Select one CORSIKA build variant.

    Parameters
    ----------
    variants : iterable of CorsikaBuildVariant
        Available variants.
    he_model : str
        High-energy hadronic interaction model.
    le_model : str
        Low-energy hadronic interaction model.
    geometry : str
        Atmosphere geometry (``flat`` or ``curved``).

    Returns
    -------
    CorsikaBuildVariant
        The unique matching variant.

    Raises
    ------
    ValueError
        If the requested combination is unavailable.
    """
    requested = (str(he_model).lower(), str(le_model).lower(), str(geometry).lower())
    for variant in variants:
        available = (
            variant.he_hadronic_model,
            variant.le_hadronic_model,
            variant.atmosphere_geometry,
        )
        if available == requested:
            return variant

    available_text = ", ".join(
        sorted(
            f"{variant.he_hadronic_model}/{variant.le_hadronic_model}/{variant.atmosphere_geometry}"
            for variant in variants
        )
    )
    raise ValueError(
        "Unsupported CORSIKA build variant "
        f"{requested[0]}/{requested[1]}/{requested[2]}. Available variants: {available_text}"
    )


def format_corsika_build_variants(variants):
    """Format installed CORSIKA build variants as a compact table.

    Parameters
    ----------
    variants : iterable of CorsikaBuildVariant
        Available variants.

    Returns
    -------
    str
        Human-readable table.
    """
    rows = [("HE model", "LE model", "geometry", "executable")]
    rows.extend(
        (
            variant.he_hadronic_model,
            variant.le_hadronic_model,
            variant.atmosphere_geometry,
            variant.executable,
        )
        for variant in variants
    )
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return "\n".join(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip()
        for row in rows
    )


def get_corsika_build_report(corsika_path=None):
    """Return a report of installed CORSIKA interaction-model variants.

    Parameters
    ----------
    corsika_path : str or Path, optional
        CORSIKA installation directory. If omitted, use ``SIMTOOLS_CORSIKA_PATH`` and then the
        simtools default path.

    Returns
    -------
    str
        Human-readable table of installed variants.

    Raises
    ------
    FileNotFoundError
        If the build manifest is missing.
    PermissionError
        If a declared executable is not executable.
    ValueError
        If the manifest is invalid or declares a missing executable.
    """
    resolved_path = Path(
        corsika_path or os.getenv("SIMTOOLS_CORSIKA_PATH") or defaults.CORSIKA_PATH
    )
    return format_corsika_build_variants(get_installed_corsika_build_variants(resolved_path))
