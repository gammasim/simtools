"""Generate Sphinx toctree index files for the generated simulation model reports."""

import json
from pathlib import Path

from simtools.utils import names

IGNORE_DIRECTORIES = {
    "_static",
    "_descriptions",
    "_images",
    "_figures",
    "_data_files",
    "scripts",
    "changes",
    "build",
}
INDEX_FILE_NAME = "index.md"
SITE_TITLE = "Simulation Models and Model Parameters"
KEEP_INDEX_MARKER = "<!-- simtools: keep-index -->"
TOCTREE_END = "```\n"


def _is_authored_index(directory):
    """Return whether the directory has an index that the generator must preserve."""
    index_file = directory / INDEX_FILE_NAME
    return index_file.exists() and KEEP_INDEX_MARKER in index_file.read_text(encoding="utf-8")


def _site_introduction():
    """Return the introduction of the site index page."""
    return (
        "This site provides comprehensive documentation for"
        " CTAO simulation models and model parameters.\n"
        "Some reports list model parameters used in a production.\n"
        "Others compare a parameter's value across different model versions.\n\n"
    )


def write_index_file(
    directory, entries, is_site_root=False, docs_path=None, simulation_models_path=None
):
    """
    Write an index.md file with a toctree listing the given entries.

    Parameters
    ----------
    directory : Path
        Directory of the index file.
    entries : list of str
        Toctree entries (document names relative to the directory).
    is_site_root : bool
        Write the site title and introduction (site root index).
    """
    title = SITE_TITLE if is_site_root else directory.name[:1].upper() + directory.name[1:]
    grouped_index = _array_element_index(
        directory,
        docs_path or directory,
        simulation_models_path=simulation_models_path,
    )
    if grouped_index is not None:
        directory.joinpath(INDEX_FILE_NAME).write_text(grouped_index, encoding="utf-8")
        return

    content = f"# {title}\n\n"
    if is_site_root:
        content += _site_introduction()
    if entries:
        content += "```{toctree}\n:maxdepth: 1\n\n" + "".join(f"{e}\n" for e in entries)
        content += TOCTREE_END
    index_file = directory / INDEX_FILE_NAME
    if _is_authored_index(directory):
        return
    index_file.write_text(content, encoding="utf-8")


def _array_element_index(directory, docs_path, simulation_models_path=None):
    """Create site and category tables for parameter and production reports."""
    if directory.name != "parameters" and not _is_model_version(directory.name):
        return None

    reports = sorted(path.stem for path in directory.glob("*.md") if path.name != INDEX_FILE_NAME)
    if not reports:
        return None

    latest_version = _latest_model_version(docs_path)
    model_version = directory.name if _is_model_version(directory.name) else None
    design_models = _load_design_models(simulation_models_path, model_version)

    categories = {
        site: {"Site": [], "Telescopes": [], "Calibration": [], "Simulation": []}
        for site in ("North", "South")
    }
    _group_reports_by_site(reports, categories)

    content = f"# {directory.name[:1].upper() + directory.name[1:]}\n\n"
    for site, site_categories in categories.items():
        content += _render_site_categories(
            site, site_categories, reports, design_models, directory, latest_version
        )

    content += "```{toctree}\n:hidden:\n:maxdepth: 1\n\n"
    content += "".join(f"{entry}\n" for entry in reports)
    content += "```\n"
    return content


def _group_reports_by_site(reports, categories):
    """Group recognized reports under their site and category."""
    for report in reports:
        report_site, category = _report_site_category(report)
        if category == "Simulation":
            for site_categories in categories.values():
                site_categories[category].append(report)
        elif category and report_site in categories:
            categories[report_site][category].append(report)


def _render_site_categories(
    site, site_categories, reports, design_models, directory, latest_version
):
    """Render one site's category headings and contents."""
    content = f"## {site}\n\n"
    for category, category_reports in site_categories.items():
        content += f"### {category}\n\n"
        content += _render_site_category(
            site,
            category,
            category_reports,
            reports,
            design_models,
            directory,
            latest_version,
        )
    return content


def _render_site_category(
    site, category, category_reports, reports, design_models, directory, latest_version
):
    """Render the content for a site's Site, Simulation, or element category."""
    if category == "Site":
        observatory = f"OBS-{site}"
        if observatory in reports or latest_version:
            target = _report_target(observatory, directory, reports, latest_version)
            return f"[{observatory}]({target})\n\n"
        return f"{observatory}\n\n"
    if category == "Simulation":
        return _render_simulation_links(directory, reports, latest_version)
    return _design_type_table(category_reports, reports, design_models, directory, latest_version)


def _render_simulation_links(directory, reports, latest_version):
    """Render the two simulation configuration links on one line."""
    simulation_links = [
        f"[{name}]({_report_target(name, directory, reports, latest_version)})"
        for name in ("configuration_corsika", "configuration_sim_telarray")
    ]
    return " · ".join(simulation_links) + "\n\n"


def _report_site_category(report):
    """Return the site and presentation category for one report name."""
    if report.startswith("configuration_"):
        return None, "Simulation"
    try:
        collection = names.get_collection_name_from_array_element_name(
            report, array_elements_only=False
        )
        site = names.get_site_from_array_element_name(report)
    except ValueError:
        return None, None
    if isinstance(site, list):
        return None, None
    category = {
        "sites": "Site",
        "telescopes": "Telescopes",
        "calibration_devices": "Calibration",
    }.get(collection)
    return site, category


def _is_model_version(name):
    """Return whether a directory name looks like a semantic model version."""
    parts = name.split(".")
    return len(parts) == 3 and all(part.isdigit() for part in parts)


def _latest_model_version(docs_path):
    """Return the newest version represented in the production documentation tree."""
    production_path = Path(docs_path) / "productions"
    versions = (
        [
            path.name
            for path in production_path.iterdir()
            if path.is_dir() and _is_model_version(path.name)
        ]
        if production_path.exists()
        else []
    )
    return max(versions, key=lambda value: tuple(map(int, value.split("."))), default=None)


def _load_design_models(simulation_models_path, model_version):
    """Read element-to-design mappings from one or all production versions."""
    if simulation_models_path is None:
        return {}
    production_root = Path(simulation_models_path) / "simulation-models" / "productions"
    if model_version:
        versions = [model_version]
    elif production_root.exists():
        versions = sorted(
            (
                path.name
                for path in production_root.iterdir()
                if path.is_dir() and _is_model_version(path.name)
            ),
            key=lambda value: tuple(map(int, value.split("."))),
        )
    else:
        versions = []
    design_models = {}
    for version in versions:
        for path in (production_root / version).glob("*.json"):
            try:
                design_models.update(
                    json.loads(path.read_text(encoding="utf-8")).get("design_model", {})
                )
            except OSError, json.JSONDecodeError:
                continue
    return design_models


def _report_target(report, directory, available_reports, latest_version):
    """Return a relative link to a report, using the latest production from parameters."""
    if report in available_reports:
        return report
    if directory.name == "parameters" and latest_version:
        return f"../productions/{latest_version}/{report}"
    return report


def _design_type_table(reports, available_reports, design_models, directory, latest_version):
    """Group reports under their declared design type."""
    groups = {}
    for report in reports:
        try:
            element_type = names.get_array_element_type_from_name(report)
            names.get_array_element_id_from_name(report)
            design = design_models.get(report)
            if design is None:
                design = (
                    report
                    if names.is_design_type(report)
                    else f"{element_type}-{names.array_element_design_types(element_type)[0]}"
                )
        except ValueError:
            design = report
        groups.setdefault(design, []).append(report)

    content = "| Design type | Array Element |\n| --- | --- |\n"
    if not groups:
        return content + "| — | — |\n\n"
    for design, elements in sorted(groups.items()):
        others = [element for element in elements if element != design]
        other_links = ", ".join(
            f"[{element}]({_report_target(element, directory, available_reports, latest_version)})"
            for element in sorted(others)
        )
        if design in available_reports or design.startswith("configuration_"):
            design_target = _report_target(design, directory, available_reports, latest_version)
            design_cell = f"[{design}]({design_target})"
        else:
            design_cell = design
        content += f"| {design_cell} | {other_links} |\n"
    return content + "\n"


def toctree_entries(directory, subdirectories):
    """Return the sorted toctree entries of a directory (markdown files, images, subdirectories)."""
    entries = []
    for path in directory.iterdir():
        if path.is_file() and path.suffix == ".md" and path.name != INDEX_FILE_NAME:
            entries.append(path.stem)
        elif path.is_file() and path.suffix == ".png":
            entries.append(path.name)
    entries.extend(f"{subdirectory.name}/index" for subdirectory in subdirectories)
    return sorted(entries)


def generate_toctree(docs_path, simulation_models_path=None):
    """
    Generate index.md files with toctrees for all directories of a documentation source tree.

    Directories used for static content, images, data files, and build output are ignored.

    Parameters
    ----------
    docs_path : Path or str
        Root directory of the Sphinx documentation source.
    """
    docs_path = Path(docs_path)
    if simulation_models_path is not None:
        simulation_models_path = Path(simulation_models_path)
    _generate_directory_toctree(docs_path, docs_path, simulation_models_path)


def _generate_directory_toctree(directory, docs_path, simulation_models_path=None):
    """Generate child indexes first and return whether this directory has an index."""
    subdirectories = []
    for path in sorted(directory.iterdir()):
        if (
            path.is_dir()
            and path.name not in IGNORE_DIRECTORIES
            and _generate_directory_toctree(path, docs_path, simulation_models_path)
        ):
            subdirectories.append(path)
    entries = toctree_entries(directory, subdirectories)
    if entries:
        write_index_file(
            directory,
            entries,
            is_site_root=directory == docs_path,
            docs_path=docs_path,
            simulation_models_path=simulation_models_path,
        )
    return bool(entries) or _is_authored_index(directory)
