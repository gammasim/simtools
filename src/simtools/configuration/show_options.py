"""Shared option-discovery helpers for selected command-line arguments."""

import argparse
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
    """Normalized output for ``--show_options``."""

    option_name: str
    help_text: str | None = None
    environment: dict = field(default_factory=dict)
    values: tuple = ()
    grouped_values: dict = field(default_factory=dict)
    notes: tuple = ()


def handle_show_options(args_dict, parser):
    """Print available values for a supported option and exit."""
    if not args_dict.get("show_options"):
        return False

    try:
        result = resolve_show_options(args_dict, parser)
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        parser.error(str(exc))

    sys.stdout.write(format_show_options_result(result) + "\n")
    parser.exit()
    return True


def resolve_show_options(args_dict, parser=None):
    """Resolve ``--show_options`` using a custom provider or parser metadata."""
    option_name = args_dict["show_options"]
    provider = _SHOW_OPTION_PROVIDERS.get(option_name)
    if provider is not None:
        return _with_argparse_help(provider(args_dict), parser)
    if parser is not None:
        return _show_argparse_option(option_name, parser)
    supported = ", ".join(sorted(_SHOW_OPTION_PROVIDERS))
    raise ValueError(
        f"Unsupported option for '--show_options': {option_name}. Supported values: {supported}."
    )


def _with_argparse_help(result, parser):
    """Attach the parser help text for a resolved option."""
    if parser is None or result.help_text is not None:
        return result
    try:
        action = _find_argparse_action(parser, result.option_name)
    except ValueError:
        return result
    return ShowOptionsResult(
        option_name=result.option_name,
        help_text=_action_help(action),
        environment=result.environment,
        values=result.values,
        grouped_values=result.grouped_values,
        notes=result.notes,
    )


def _show_argparse_option(option_name, parser):
    """Build a result from an argparse action when no custom provider exists."""
    action = _find_argparse_action(parser, option_name)
    values = tuple(str(value) for value in action.choices) if action.choices is not None else ()
    notes = () if values else ("Argparse does not define a finite set of available values.",)
    return ShowOptionsResult(
        option_name=action.dest,
        help_text=_action_help(action),
        values=values,
        notes=notes,
    )


def _find_argparse_action(parser, option_name):
    """Find a parser action by destination name or hyphenated destination name."""
    destination = option_name.removeprefix("--").replace("-", "_")
    for action in parser._actions:  # pylint: disable=protected-access
        if action.dest == destination:
            return action
    raise ValueError(f"Unknown command-line option for --show_options: {option_name}.")


def _action_help(action):
    """Return visible help text from an argparse action."""
    if action.help in (None, argparse.SUPPRESS):
        return None
    return action.help


def format_show_options_result(result):
    """Format one ``ShowOptionsResult``."""
    lines = [f"Option: {result.option_name}"]

    if result.help_text:
        lines.extend(["", "Help:", f"  {result.help_text}"])

    if result.environment:
        lines.extend(["", "Environment:"])
        lines.extend(f"  {label}: {value}" for label, value in result.environment.items())

    if result.notes:
        lines.extend(["", "Notes:"])
        lines.extend(f"  {note}" for note in result.notes)

    if result.grouped_values or result.values:
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


def _show_corsika_hadronic_transition_energy(args_dict):
    """Show CORSIKA HILOW defaults recorded for installed build variants."""
    variants, resolved_path = _get_corsika_variants(args_dict)
    grouped_values = {}
    missing_variants = []
    for variant in variants:
        variant_name = (
            f"{variant.he_hadronic_model}/{variant.le_hadronic_model}/{variant.atmosphere_geometry}"
        )
        transition_energy = getattr(variant, "hadronic_transition_energy_default_gev", None)
        if transition_energy is None:
            missing_variants.append(variant_name)
        else:
            grouped_values[variant_name] = (f"{transition_energy:g} GeV",)

    notes = ()
    if missing_variants:
        notes = (
            "The installed CORSIKA build metadata does not declare HILOW for: "
            + ", ".join(missing_variants)
            + ".",
        )
    return ShowOptionsResult(
        option_name="corsika_hadronic_transition_energy",
        environment=_get_corsika_environment(resolved_path, variants),
        grouped_values=grouped_values,
        notes=notes,
    )


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
        raise ValueError(f"'--show_options {option_name}' requires '--model_version'.")
    if isinstance(model_version, list):
        if len(model_version) != 1:
            raise ValueError(
                f"'--show_options {option_name}' requires exactly one '--model_version' value."
            )
        return model_version[0]
    return model_version


_SHOW_OPTION_PROVIDERS = {
    "array_layout_name": _show_array_layout_names,
    "corsika_he_interaction": _show_corsika_he_interaction,
    "corsika_le_interaction": _show_corsika_le_interaction,
    "model_version": _show_model_versions,
    "corsika_hadronic_transition_energy": _show_corsika_hadronic_transition_energy,
    "primary": _show_primary,
    "site": _show_sites,
}
