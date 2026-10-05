"""Generate markdown summaries for production descriptions."""

from pathlib import Path

from packaging.version import InvalidVersion, Version

from simtools.io import ascii_handler


def collect_production_descriptions(data_path=None, model_reader=None):
    """Collect production versions and descriptions from simulation-models info files.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to the simulation-models repository root.
    model_reader : SimulationModelReader, optional
        Reader for a filesystem or Git simulation-model source.

    Returns
    -------
    list[tuple]
        List with ``(model_version, description)`` pairs sorted by version string.
    """
    if model_reader is not None:
        return model_reader.get_production_descriptions()
    if data_path is None:
        raise ValueError("A model path or model reader is required.")

    productions_path = Path(data_path) / "simulation-models" / "productions"

    def _version_sort_key(path):
        try:
            return (0, Version(path.parent.name))
        except InvalidVersion:
            return (1, path.parent.name)

    info_files = sorted(
        set(productions_path.glob("*/info.yaml")) | set(productions_path.glob("*/info.yml")),
        key=_version_sort_key,
    )

    descriptions = []
    for info_file in info_files:
        info = ascii_handler.collect_data_from_file(info_file)
        description = str(info.get("description", "")).replace("\n", " ").strip()
        model_version = str(info.get("model_version", info_file.parent.name))
        descriptions.append((model_version, description))

    return descriptions


def write_production_summary_markdown(data_path=None, output_file=None, model_reader=None):
    """Write a markdown table with production versions and short descriptions.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to the simulation-models repository root.
    output_file : str or Path, optional
        Markdown file to write.
    model_reader : SimulationModelReader, optional
        Reader for a filesystem or Git simulation-model source.
    """
    if output_file is None:
        raise ValueError("An output file is required.")

    lines = [
        "# Descriptions of productions",
        "",
        "| Production Model Version | Short Description                     |",
        "|---------------------------|---------------------------------------|",
    ]

    for model_version, description in collect_production_descriptions(
        data_path=data_path, model_reader=model_reader
    ):
        escaped_description = description.replace("|", "\\|")
        lines.append(f"| {model_version} | {escaped_description} |")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
