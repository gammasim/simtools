from pathlib import Path

from simtools.reporting import docs_toctree

INDEX = "index.md"


def _create_docs_tree(docs):
    (docs / "parameters").mkdir(parents=True)
    (docs / "productions" / "6.0.0").mkdir(parents=True)
    (docs / "productions" / "6.0.0" / "_data_files").mkdir()
    (docs / "_static").mkdir()
    (docs / "parameters" / "LSTN-01.md").write_text("# LSTN-01\n")
    (docs / "parameters" / "LSTN-design.md").write_text("# LSTN-design\n")
    (docs / "productions" / "6.0.0" / "OBS-North.md").write_text("# OBS-North\n")
    (docs / "productions" / "6.0.0" / "_data_files" / "data.md").write_text("# data\n")
    (docs / "productions" / "layout.png").write_bytes(b"png")
    (docs / "_static" / "ignored.md").write_text("# ignored\n")


def test_generate_toctree(tmp_test_directory):
    docs = Path(tmp_test_directory) / "docs"
    _create_docs_tree(docs)
    (docs / docs_toctree.PRODUCTION_DESCRIPTIONS_FILE_NAME).write_text(
        "# Title\n| table |\n", encoding="utf-8"
    )

    docs_toctree.generate_toctree(docs)

    root_index = (docs / INDEX).read_text(encoding="utf-8")
    assert root_index.startswith(f"# {docs_toctree.SITE_TITLE}\n")
    assert "| table |" in root_index
    assert "# Title" not in root_index
    assert "_descriptions/description\n" in root_index
    assert "parameters/index\nproduction_version_descriptions\nproductions/index\n" in root_index

    parameters_index = (docs / "parameters" / INDEX).read_text(encoding="utf-8")
    assert (
        parameters_index
        == "# Parameters\n\n```{toctree}\n:maxdepth: 1\n\nLSTN-01\nLSTN-design\n```\n"
    )

    productions_index = (docs / "productions" / INDEX).read_text(encoding="utf-8")
    assert "6.0.0/index\nlayout.png\n" in productions_index
    assert "OBS-North" in (docs / "productions" / "6.0.0" / INDEX).read_text(encoding="utf-8")
    assert not (docs / "productions" / "6.0.0" / "_data_files" / INDEX).exists()
    assert not (docs / "_static" / INDEX).exists()


def test_generate_toctree_without_production_descriptions(tmp_test_directory):
    docs = Path(tmp_test_directory) / "docs"
    _create_docs_tree(docs)

    docs_toctree.generate_toctree(docs)

    root_index = (docs / INDEX).read_text(encoding="utf-8")
    assert "_descriptions/description" not in root_index
    assert "parameters/index\nproductions/index\n" in root_index


def test_generate_toctree_skips_directories_without_entries(tmp_test_directory):
    docs = Path(tmp_test_directory) / "docs"
    (docs / "empty").mkdir(parents=True)

    docs_toctree.generate_toctree(docs)

    assert not (docs / "empty" / INDEX).exists()
    assert (docs / INDEX).read_text(encoding="utf-8").endswith("empty/index\n```\n")
