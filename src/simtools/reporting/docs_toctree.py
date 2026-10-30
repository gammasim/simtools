"""Generate Sphinx toctree index files for the generated simulation model reports."""

from pathlib import Path

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


def write_index_file(directory, entries, is_site_root=False):
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


def generate_toctree(docs_path):
    """
    Generate index.md files with toctrees for all directories of a documentation source tree.

    Directories used for static content, images, data files, and build output are ignored.

    Parameters
    ----------
    docs_path : Path or str
        Root directory of the Sphinx documentation source.
    """
    docs_path = Path(docs_path)
    _generate_directory_toctree(docs_path, docs_path)


def _generate_directory_toctree(directory, docs_path):
    """Generate child indexes first and return whether this directory has an index."""
    subdirectories = []
    for path in sorted(directory.iterdir()):
        if (
            path.is_dir()
            and path.name not in IGNORE_DIRECTORIES
            and _generate_directory_toctree(path, docs_path)
        ):
            subdirectories.append(path)
    entries = toctree_entries(directory, subdirectories)
    if entries:
        write_index_file(directory, entries, is_site_root=directory == docs_path)
    return bool(entries) or _is_authored_index(directory)
