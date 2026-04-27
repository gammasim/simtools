"""Generate markdown summaries for production descriptions."""

from pathlib import Path

from simtools.io import ascii_handler


def collect_production_descriptions(data_path):
    """Collect production versions and descriptions from simulation-models info files.

    Parameters
    ----------
    data_path : str or Path
        Path to the simulation-models repository root.

    Returns
    -------
    list[tuple]
        List with ``(model_version, description)`` pairs sorted by version string.
    """
    productions_path = Path(data_path) / "simulation-models" / "productions"

    info_files = sorted(
        set(productions_path.glob("*/info.yaml")) | set(productions_path.glob("*/info.yml")),
        key=lambda path: path.parent.name,
    )

    descriptions = []
    for info_file in info_files:
        info = ascii_handler.collect_data_from_file(info_file)
        description = str(info.get("description", "")).replace("\n", " ").strip()
        model_version = str(info.get("model_version", info_file.parent.name))
        descriptions.append((model_version, description))

    return descriptions


def write_production_summary_markdown(data_path, output_file):
    """Write a markdown table with production versions and short descriptions.

    Parameters
    ----------
    data_path : str or Path
        Path to the simulation-models repository root.
    output_file : str or Path
        Markdown file to write.
    """
    lines = [
        "# Descriptions of productions",
        "",
        "| Production Model Version | Short Description                     |",
        "|---------------------------|---------------------------------------|",
    ]

    for model_version, description in collect_production_descriptions(data_path):
        escaped_description = description.replace("|", "\\|")
        lines.append(f"| {model_version} | {escaped_description} |")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
