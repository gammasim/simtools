from pathlib import Path

import pytest

from simtools.reporting import docs_toctree

INDEX = "index.md"
NESTED = "nested"
ENCODING = "utf-8"
PARAMETERS = "parameters"
MODEL_VERSION = "6.0.0"
DATA_FILES = "_data_files"
STATIC = "_static"
IGNORED_MD = "ignored.md"
EMPTY = "empty"
PRODUCTIONS = "productions"


def _create_docs_tree(docs):
    (docs / PARAMETERS).mkdir(parents=True)
    (docs / PRODUCTIONS / MODEL_VERSION).mkdir(parents=True)
    (docs / PRODUCTIONS / MODEL_VERSION / DATA_FILES).mkdir()
    (docs / STATIC).mkdir()
    (docs / PARAMETERS / "LSTN-01.md").write_text("# LSTN-01\n")
    (docs / PARAMETERS / "LSTN-design.md").write_text("# LSTN-design\n")
    (docs / PRODUCTIONS / MODEL_VERSION / "OBS-North.md").write_text("# OBS-North\n")
    (docs / PRODUCTIONS / MODEL_VERSION / DATA_FILES / "data.md").write_text("# data\n")
    (docs / PRODUCTIONS / "layout.png").write_bytes(b"png")
    (docs / STATIC / IGNORED_MD).write_text("# ignored\n")


def test_generate_toctree(tmp_test_directory):
    docs = Path(tmp_test_directory) / "docs"
    _create_docs_tree(docs)
    docs_toctree.generate_toctree(docs)

    root_index = (docs / INDEX).read_text(encoding=ENCODING)
    assert root_index.startswith(f"# {docs_toctree.SITE_TITLE}\n")
    assert "parameters/index\nproductions/index\n" in root_index

    parameters_index = (docs / PARAMETERS / INDEX).read_text(encoding=ENCODING)
    assert (
        parameters_index
        == "# Parameters\n\n```{toctree}\n:maxdepth: 1\n\nLSTN-01\nLSTN-design\n```\n"
    )

    productions_index = (docs / PRODUCTIONS / INDEX).read_text(encoding=ENCODING)
    assert "6.0.0/index\nlayout.png\n" in productions_index
    assert "OBS-North" in (docs / PRODUCTIONS / MODEL_VERSION / INDEX).read_text(encoding=ENCODING)
    assert not (docs / PRODUCTIONS / MODEL_VERSION / DATA_FILES / INDEX).exists()
    assert not (docs / STATIC / INDEX).exists()


def test_generate_toctree_includes_authored_only_directory(tmp_test_directory):
    docs = Path(tmp_test_directory) / "docs"
    introduction = docs / "introduction"
    introduction.mkdir(parents=True)
    (introduction / INDEX).write_text(
        f"{docs_toctree.KEEP_INDEX_MARKER}\n# Introduction\n", encoding=ENCODING
    )

    docs_toctree.generate_toctree(docs)

    assert "introduction/index" in (docs / INDEX).read_text(encoding=ENCODING)


@pytest.mark.parametrize("ignored_directory", sorted(docs_toctree.IGNORE_DIRECTORIES))
def test_generate_toctree_ignores_directories(tmp_test_directory, ignored_directory):
    docs = Path(tmp_test_directory) / "docs"
    ignored = docs / ignored_directory
    ignored.mkdir(parents=True)
    (ignored / IGNORED_MD).write_text("# Ignored\n", encoding=ENCODING)

    docs_toctree.generate_toctree(docs)

    assert not (ignored / INDEX).exists()
    assert not (docs / INDEX).exists()
