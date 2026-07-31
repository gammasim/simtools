"""Shared option-discovery helpers for selected command-line arguments."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from simtools.configuration import defaults
from simtools.corsika.build_options import get_installed_corsika_build_variants
from simtools.corsika.primary_particle import PrimaryParticle
from simtools.db.db_handler import DatabaseHandler
from simtools.dependencies import get_corsika_version
from simtools.model.site_model import SiteModel
from simtools.utils import names


@dataclass(frozen=True)
class ShowOptionsResult:
    """Normalized output for ``--show-options``."""

    option_name: str
    environment: dict[str, str] = field(default_factory=dict)
    values: tuple[str, ...] = ()
    grouped_values: dict[str, tuple[str, ...]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


def handle_show_options(args_dict, parser):
    """Print available values for a supported option and exit."""
    if not args_dict.get("show_options"):
        return False

    try:
        result = resolve_show_options(args_dict)
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        parser.error(str(exc))

    sys.stdout.write(format_show_options_result(result) + "\n")
    parser.exit()
    return True


def resolve_show_options(args_dict):
    """Resolve ``--show-options`` into a printable result."""
    option_name = args_dict["show_options"]
    try:
        provider = _SHOW_OPTION_PROVIDERS[option_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_SHOW_OPTION_PROVIDERS))
        raise ValueError(
            f"Unsupported option for '--show-options': {option_name}. "
            f"Supported values: {supported}."
        ) from exc
    return provider(args_dict)


def format_show_options_result(result):
    """Format one ``ShowOptionsResult``."""
    lines = [f"Option: {result.option_name}"]

    if result.environment:
        lines.extend(["", "Environment:"])
        lines.extend(f"  {label}: {value}" for label, value in result.environment.items())

    if result.notes:
        lines.extend(["", "Notes:"])
        lines.extend(f"  {note}" for note in result.notes)

    lines.extend(["", "Available values:"])
    if result.grouped_values:
        for group_name, values in result.grouped_values.items():
            lines.append(f"  {group_name}:")
            lines.extend(f"    {value}" for value in values)
    else:
        lines.extend(f"  {value}" for value in result.values)
    return "\n".join(lines)


def _show_primary(_args_dict):
    return ShowOptionsResult(
        option_name="primary",
        values=tuple(PrimaryParticle.available_particle_names()),
    )


def _show_sites(_args_dict):
    return ShowOptionsResult(option_name="site", values=tuple(names.site_names()))


def _show_model_versions(_args_dict):
    return ShowOptionsResult(
        option_name="model_version",
        values=tuple(DatabaseHandler().get_model_versions()),
    )


def _show_array_layout_names(args_dict):
    model_version = _get_single_model_version(args_dict, "array_layout_name")
    site = args_dict.get("site")
    sites = [site] if site else sorted(names.site_names())
    grouped_values = {
        current_site: tuple(
            SiteModel(site=current_site, model_version=model_version).get_list_of_array_layouts()
        )
        for current_site in sites
    }
    if site:
        return ShowOptionsResult(
            option_name="array_layout_name",
            grouped_values=grouped_values,
        )
    return ShowOptionsResult(
        option_name="array_layout_name",
        grouped_values=grouped_values,
        notes=("Provide --site to limit array layouts to one site.",),
    )


def _show_corsika_he_interaction(args_dict):
    return _show_corsika_interaction(args_dict, interaction_level="high")


def _show_corsika_le_interaction(args_dict):
    return _show_corsika_interaction(args_dict, interaction_level="low")


def _show_corsika_interaction(args_dict, interaction_level):
    variants, resolved_path = _get_corsika_variants(args_dict)
    if interaction_level == "high":
        option_name = "corsika_he_interaction"
        values = sorted({variant.he_hadronic_model for variant in variants})
    else:
        option_name = "corsika_le_interaction"
        values = sorted({variant.le_hadronic_model for variant in variants})
    return ShowOptionsResult(
        option_name=option_name,
        environment=_get_corsika_environment(resolved_path, variants),
        values=tuple(values),
    )


def _get_corsika_variants(args_dict):
    resolved_path = _resolve_corsika_path(args_dict)
    return get_installed_corsika_build_variants(resolved_path), resolved_path


def _get_corsika_environment(corsika_path, variants):
    environment = {"path": str(corsika_path)}
    corsika_version = get_corsika_version()
    if corsika_version:
        environment["CORSIKA version"] = corsika_version
    executables = sorted(
        str((Path(corsika_path) / variant.executable).resolve()) for variant in variants
    )
    if len(executables) == 1:
        environment["executable"] = executables[0]
    else:
        environment["executables"] = ", ".join(executables)
    return environment


def _resolve_corsika_path(args_dict):
    return Path(
        args_dict.get("corsika_path") or os.getenv("SIMTOOLS_CORSIKA_PATH") or defaults.CORSIKA_PATH
    )


def _get_single_model_version(args_dict, option_name):
    model_version = args_dict.get("model_version")
    if model_version is None:
        raise ValueError(f"'--show-options {option_name}' requires '--model_version'.")
    if isinstance(model_version, list):
        if len(model_version) != 1:
            raise ValueError(
                f"'--show-options {option_name}' requires exactly one '--model_version' value."
            )
        return model_version[0]
    return model_version


_SHOW_OPTION_PROVIDERS = {
    "array_layout_name": _show_array_layout_names,
    "corsika_he_interaction": _show_corsika_he_interaction,
    "corsika_le_interaction": _show_corsika_le_interaction,
    "model_version": _show_model_versions,
    "primary": _show_primary,
    "site": _show_sites,
}
