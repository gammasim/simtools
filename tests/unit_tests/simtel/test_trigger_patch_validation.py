"""Tests for planned trigger stimuli and complete event accounting."""

import json
from pathlib import Path

import numpy as np
import pytest

from simtools.simtel import trigger_patch_validation as validation

NATIVE_CAMERA = (
    "Rotate 90\nPixel 0 1 0 0\nPixel 1 1 10 0\nPixel 2 1 0 20\n"
    "MajorityTrigger 2 of +0 1[2]\nAnalogSumTrigger * of 0 1 2\n"
)


def _camera(tmp_test_directory):
    path = Path(tmp_test_directory) / "camera.dat"
    path.write_text(NATIVE_CAMERA, encoding=validation.UTF_8)
    return path


def test_bounded_planning_deduplicates_and_preserves_group_associations(tmp_test_directory):
    groups, pixels, _rotation = validation.read_camera(_camera(tmp_test_directory))
    rows = validation.plan_patterns(groups, pixels, 2, 2, False)
    assert len(rows) == 2
    assert json.loads(rows[0][validation.INPUT_PIXELS]) == [0, 1]
    assert json.loads(rows[0][validation.SOURCE_GROUPS]) == [0, 1]
    capped = validation.plan_patterns(groups, pixels, 2, 2, False, max_events=1)
    assert len(capped) == 1


def test_default_whole_groups_have_no_assumed_pixel_count(tmp_test_directory):
    groups, pixels, _rotation = validation.read_camera(_camera(tmp_test_directory))
    rows = validation.plan_patterns(groups, pixels, None, 1, False)
    assert [json.loads(row[validation.INPUT_PIXELS]) for row in rows] == [[0, 1], [0, 1, 2]]
    central = validation.plan_patterns(groups, pixels, None, 1, False, max_radius=0.11)
    assert len(central) == 1
    assert json.loads(central[0][validation.SOURCE_GROUPS]) == [0]


def test_radius_and_slave_handling(tmp_test_directory):
    groups, pixels, _rotation = validation.read_camera(_camera(tmp_test_directory))
    rows = validation.plan_patterns(groups, pixels, 2, 1, True, max_radius=0.21)
    assert json.loads(rows[0][validation.INPUT_PIXELS]) == [0, 1, 2]
    with pytest.raises(ValueError, match="No illumination patterns selected"):
        validation.plan_patterns(groups, pixels, 2, 1, False, max_radius=0.01)


def test_exhaustive_combinations_are_explicit(tmp_test_directory):
    groups, pixels, _rotation = validation.read_camera(_camera(tmp_test_directory))
    rows = validation.plan_patterns(groups, pixels, 2, 1, False, exhaustive=True)
    assert len(rows) == 3


@pytest.mark.parametrize("definition", ["MajorityTrigger 2", "MajorityTrigger 2 of invalid"])
def test_malformed_native_groups_fail_before_simulation(tmp_test_directory, definition):
    path = _camera(tmp_test_directory)
    path.write_text(definition, encoding=validation.UTF_8)
    with pytest.raises(ValueError, match="Malformed native trigger"):
        validation.read_camera(path)


def test_exhaustive_cap_does_not_materialize_the_full_combination_space():
    group = {
        validation.MASTERS: list(range(60)),
        validation.REQUIRED: [],
        validation.SLAVES: {pixel: [] for pixel in range(60)},
    }
    pixels = dict.fromkeys(range(60), (0, 0))
    rows = validation.plan_patterns([group], pixels, 30, 1, False, exhaustive=True, max_events=2)
    assert len(rows) == 2
    assert all(len(json.loads(row[validation.INPUT_PIXELS])) == 30 for row in rows)


def test_retained_readout_is_not_evidence_of_a_trigger():
    row = {validation.INPUT_PIXELS: "[0]", validation.MAPPING_STATUS: validation.INCONCLUSIVE}
    validation._record_event(
        row,
        {
            validation.EVENT_ID: 100,
            validation.TRIGGER_INFORMATION: {validation.TRIGGERED_TELESCOPES: []},
            validation.TELESCOPE_EVENTS: {1: {validation.HEADER: {validation.LIST_TRGSECT: [0]}}},
        },
        1,
    )
    assert row[validation.STATE] == validation.NON_TRIGGERED


def test_duplicates_and_unknown_event_ids_do_not_replace_missing_trials(tmp_test_directory):
    groups, pixels, _rotation = validation.read_camera(_camera(tmp_test_directory))
    patterns = validation.plan_patterns(groups, pixels, 2, 2, False)
    group_rows = [
        {
            validation.PLANNED: 0,
            validation.OBSERVED: 0,
            validation.TRIGGERED: 0,
            validation.INTENDED_SECTOR_FIRED: 0,
        }
        for _ in groups
    ]
    events = [
        {validation.EVENT_ID: i * 100, validation.MC_EVENT: {validation.SHOWER_NUMBER: i}}
        for i in (1, 1, 5)
    ]
    rows, unknown, duplicates, _samples = validation.account_events(events, 1, patterns, group_rows)
    assert unknown == [500]
    assert duplicates == [1]
    assert rows[1][validation.STATE] == validation.MISSING


def test_single_mirror_stimuli_invert_camera_rotation_and_y_flip(tmp_test_directory):
    source = _camera(tmp_test_directory)
    groups, pixels, _rotation = validation.read_camera(source)
    rows = validation.plan_patterns(groups, pixels, 2, 1, False)
    destination = Path(tmp_test_directory) / "stimuli.dat"
    validation.write_stimulus_camera(source, destination, rows, mirror_class=0)
    generated_groups, positions, _rotation = validation.read_camera(destination)
    assert positions[1] == pytest.approx((0, -10), abs=1e-10)
    assert generated_groups[0][validation.REQUIRED] == [0, 1]
    assert len(generated_groups) == len(rows)


def test_header_comparison_detects_wrong_membership_and_kind(tmp_test_directory):
    groups, _pixels, _rotation = validation.read_camera(_camera(tmp_test_directory))
    organization = {
        "n_sectors": 2,
        validation.SECTORS: [[0, 1], [0, 1], [0, 1]],
        validation.SECTOR_TYPE: [0, 1],
        "sector_threshold": [2, 3],
        "sector_pixel_threshold": [0, 0],
    }
    rows = validation.build_group_rows(groups, organization, [True, True])
    assert all(row[validation.HEADER_MATCH] for row in rows)
    organization[validation.SECTOR_TYPE][0] = 2
    organization[validation.SECTORS][0] = [1]
    rows = validation.build_group_rows(groups, organization, [True, False])
    assert not rows[0][validation.HEADER_MATCH]
    assert rows[1][validation.MODEL_MATCH] == validation.MISMATCH


def test_non_triggers_and_missing_events_have_separate_denominators(tmp_test_directory):
    groups, pixels, _rotation = validation.read_camera(_camera(tmp_test_directory))
    patterns = validation.plan_patterns(groups, pixels, 2, 3, False)
    group_rows = [
        {
            validation.PLANNED: 0,
            validation.OBSERVED: 0,
            validation.TRIGGERED: 0,
            validation.INTENDED_SECTOR_FIRED: 0,
        }
        for _ in groups
    ]
    events = [
        {validation.EVENT_ID: 100, validation.MC_EVENT: {validation.SHOWER_NUMBER: 1}},
        {
            validation.EVENT_ID: 200,
            validation.MC_EVENT: {validation.SHOWER_NUMBER: 2},
            validation.TELESCOPE_EVENTS: {
                1: {validation.HEADER: {validation.LIST_TRGSECT: np.array([0])}}
            },
        },
    ]
    rows, unexpected, duplicates, samples = validation.account_events(
        events, 1, patterns, group_rows
    )
    assert [row[validation.STATE] for row in rows] == [
        validation.NON_TRIGGERED,
        validation.TRIGGERED,
        validation.MISSING,
    ]
    assert not unexpected
    assert not duplicates
    assert len(samples) == 2
    assert group_rows[0][validation.PLANNED] == 3
    assert group_rows[0][validation.OBSERVED] == 2
    assert group_rows[0][validation.INTENDED_SECTOR_FIRED] == 1


@pytest.mark.parametrize("limit", [0, 1, 3, 8])
def test_event_page_limit_keeps_repeated_outcomes_and_accounts_for_all_trials(
    tmp_test_directory, limit
):
    groups, pixels, _rotation = validation.read_camera(_camera(tmp_test_directory))
    patterns = validation.plan_patterns(groups, pixels, 2, 4, False)
    group_rows = [
        dict.fromkeys(
            [
                validation.PLANNED,
                validation.OBSERVED,
                validation.TRIGGERED,
                validation.INTENDED_SECTOR_FIRED,
            ],
            0,
        )
        for _ in groups
    ]
    events = [
        {validation.EVENT_ID: i * 100, validation.MC_EVENT: {validation.SHOWER_NUMBER: i}}
        for i in range(1, 5)
    ]
    rows, unknown, duplicates, samples = validation.account_events(
        events, 1, patterns, group_rows, sample_limit=limit
    )
    assert len(rows) == 4
    assert not unknown
    assert not duplicates
    assert samples == events[:limit]
    assert group_rows[0][validation.OBSERVED] == 4


def test_model_comparison_uses_roles_and_detects_removed_members(tmp_test_directory):
    groups, _pixels, _rotation = validation.read_camera(_camera(tmp_test_directory))
    group_table = [
        {
            validation.KIND: "majority",
            validation.USE_DEFAULT_MULTIPLICITY: False,
            validation.MULTIPLICITY: 2,
        }
    ]
    members = [
        {
            validation.GROUP_ID: 0,
            validation.MEMBER_ORDER: 0,
            validation.PIXEL_ORDER: 0,
            validation.PIXEL_ID: 0,
            validation.REQUIRED: True,
        },
        {
            validation.GROUP_ID: 0,
            validation.MEMBER_ORDER: 1,
            validation.PIXEL_ORDER: 0,
            validation.PIXEL_ID: 1,
            validation.REQUIRED: False,
        },
        {
            validation.GROUP_ID: 0,
            validation.MEMBER_ORDER: 1,
            validation.PIXEL_ORDER: 1,
            validation.PIXEL_ID: 2,
            validation.REQUIRED: False,
        },
    ]
    matches, count_matches = validation.compare_model(groups[:1], group_table, members)
    assert matches == [True]
    assert count_matches
    assert validation.compare_model(groups[:1], group_table, list(reversed(members)))[0] == [True]
    matches, count_matches = validation.compare_model(groups, group_table, members[:-1])
    assert matches == [False, False]
    assert not count_matches
