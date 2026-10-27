"""Generate markdown summaries for production descriptions."""

from pathlib import Path

from packaging.version import Version

from simtools.io import ascii_handler
from simtools.model_repository import files


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
        List with ``(model_version, description)`` pairs from the model source.
    """
    if model_reader is not None:
        return model_reader.get_production_descriptions()
    if data_path is None:
        raise ValueError("A model path or model reader is required.")

    productions_path = Path(data_path) / "simulation-models" / "productions"

    info_files = set(productions_path.glob("*/info.yaml")) | set(
        productions_path.glob("*/info.yml")
    )
    return files.collect_production_descriptions(info_files, ascii_handler.collect_data_from_file)


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

    descriptions = sorted(
        collect_production_descriptions(data_path=data_path, model_reader=model_reader),
        key=lambda item: Version(item[0]),
        reverse=True,
    )
    lines = [
        "<!-- simtools: keep-index -->",
        "# Productions",
        "",
        "Production versions define sets of model parameters used in simulations.",
        "Select a version to browse its array and simulation reports.",
        "",
        "| Production version | Description |",
        "| --- | --- |",
    ]

    for model_version, description in descriptions:
        escaped_description = description.replace("|", "\\|")
        lines.append(f"| [{model_version}]({model_version}/index.md) | {escaped_description} |")

    lines.extend(["", "```{toctree}", ":hidden:", ":maxdepth: 1", ""])
    lines.extend(f"{model_version}/index" for model_version, _ in descriptions)
    lines.extend(["```", ""])

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
