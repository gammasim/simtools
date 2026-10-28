"""Tests for shared production document processing."""

from pathlib import PurePosixPath

from simtools.model_repository.files import (
    collect_production_descriptions,
    read_production_tables_from_documents,
)


def test_collect_production_descriptions_reads_and_sorts_info_documents():
    """Production descriptions are sorted by semantic model version."""
    records = {
        "1.0.0": {"description": " First\nrelease ", "model_version": "1.0.0"},
        "2.0.0": {"description": "Second"},
    }
    paths = [
        PurePosixPath("simulation-models/productions/2.0.0/info.yml"),
        PurePosixPath("simulation-models/productions/1.0.0/info.yml"),
    ]

    result = collect_production_descriptions(paths, lambda path: records[path.parent.name])

    assert result == [("1.0.0", "First release"), ("2.0.0", "Second")]


def test_read_production_tables_from_documents_merges_collection_and_version():
    """Documents combine parameter and design metadata into a production table."""
    table = read_production_tables_from_documents(
        "1.0.0",
        [
            (
                "1.0.0",
                "simulation-models/productions/1.0.0/LSTN-01.json",
                {
                    "parameters": {"LSTN-01": {"camera_body_diameter": "1.0.0"}},
                    "design_model": {"LSTN-01": "LSTN-design"},
                },
            )
        ],
    )

    assert table["telescopes"]["model_version"] == "1.0.0"
    assert table["telescopes"]["parameters"] == {"LSTN-01": {"camera_body_diameter": "1.0.0"}}
    assert table["telescopes"]["design_model"] == {"LSTN-01": "LSTN-design"}
