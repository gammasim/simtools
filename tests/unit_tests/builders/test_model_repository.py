"""Tests for schema-driven model repository builders."""

import json
import math

from simtools.model_repository.reader import SimulationModelReader
from tests.unit_tests.builders.model_repository import (
    model_reader,
    model_repository_files,
    write_model_repository,
)


def test_model_reader_returns_element_and_site_parameters():
    """The memory reader resolves generated telescope and site parameters."""
    reader = model_reader()

    for element, site in (("LSTN-01", "North"), ("MSTS-01", "South")):
        telescope = reader.get_model_parameters(
            site, element, "telescopes", "6.0.0", ["camera_body_diameter"]
        )
        site_parameters = reader.get_model_parameters(
            site, f"OBS-{site}", "sites", "6.0.0", ["corsika_observation_level"]
        )

        assert telescope["camera_body_diameter"]["instrument"] == element
        assert site_parameters["corsika_observation_level"]["site"] == site


def test_model_reader_exposes_overrides():
    """Overrides are stored in generated parameter records."""
    reader = model_reader(
        overrides={
            ("LSTN-01", "camera_body_diameter"): 1.5,
            ("North", "corsika_observation_level"): 2000.0,
        }
    )

    parameters = reader.get_model_parameters(
        "North", "LSTN-01", "telescopes", "6.0.0", ["camera_body_diameter"]
    )

    assert math.isclose(parameters["camera_body_diameter"]["value"], 1.5)
    site_parameters = reader.get_model_parameters(
        "North", "OBS-North", "sites", "6.0.0", ["corsika_observation_level"]
    )
    assert math.isclose(site_parameters["corsika_observation_level"]["value"], 2000.0)


def test_directory_repository_reads_the_same_parameters(tmp_test_directory):
    """The directory-backed repository exposes the same generated values."""
    files = model_repository_files()
    write_model_repository(tmp_test_directory)
    memory_reader = SimulationModelReader.from_memory(files)
    directory_reader = SimulationModelReader.from_files(tmp_test_directory)

    assert directory_reader.get_model_parameters(
        "South", "MSTS-01", "telescopes", "6.0.0", ["camera_body_diameter"]
    ) == memory_reader.get_model_parameters(
        "South", "MSTS-01", "telescopes", "6.0.0", ["camera_body_diameter"]
    )
    assert (
        json.loads(
            (tmp_test_directory / "simulation-models/productions/6.0.0/LSTN-01.json").read_text(
                encoding="utf-8"
            )
        )["production_table_name"]
        == "LSTN-01"
    )
