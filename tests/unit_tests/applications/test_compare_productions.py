import pytest

import simtools.sim_events.production_comparison as production_comparison


def test_parse_production_arguments_accepts_single_production(mocker):
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    descriptors = production_comparison.parse_production_arguments([["baseline", "base.h5"]])

    assert len(descriptors) == 1
    assert descriptors[0].label == "baseline"
    assert descriptors[0].trigger_histogram_files == ["base.h5"]


def test_parse_production_arguments_resolves_flattened_pairs(mocker):
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    descriptors = production_comparison.parse_production_arguments(
        ["baseline", "base_*.h5", "candidate", "cand_*.h5"]
    )

    assert [descriptor.label for descriptor in descriptors] == ["baseline", "candidate"]

    assert descriptors[0].trigger_histogram_files == ["base_*.h5"]
    assert descriptors[1].trigger_histogram_files == ["cand_*.h5"]


def test_parse_production_arguments_rejects_duplicate_labels(mocker):
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    with pytest.raises(ValueError, match="labels must be unique"):
        production_comparison.parse_production_arguments([["same", "a.h5"], ["same", "b.h5"]])


@pytest.mark.parametrize(
    ("arguments", "error_match"),
    [
        ([], "At least one production is required"),
        (["baseline", "base.h5", "dangling"], "label/file pairs"),
        ([["baseline", "  ,   "]], "has no trigger_histogram_file pattern"),
        ([["baseline", "a.h5"], ["candidate", 1]], "label/file pairs"),
    ],
)
def test_parse_production_arguments_error_paths(mocker, arguments, error_match):
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    with pytest.raises(ValueError, match=error_match):
        production_comparison.parse_production_arguments(arguments)


def test_parse_production_arguments_rejects_unresolved_patterns(mocker):
    mocker.patch("simtools.sim_events.production_comparison.resolve_file_patterns", return_value=[])

    with pytest.raises(ValueError, match="does not resolve to any files"):
        production_comparison.parse_production_arguments([["baseline", "missing_*.h5"]])


def test_parse_production_arguments_accepts_nested_flattened_strings(mocker):
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    descriptors = production_comparison.parse_production_arguments(
        [["baseline", "a.h5", "candidate", "b.h5"]]
    )

    assert [descriptor.label for descriptor in descriptors] == ["baseline", "candidate"]
