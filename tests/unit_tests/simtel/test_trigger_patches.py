"""Tests for camera trigger-patch validation."""

import pytest

from simtools.simtel.trigger_patches import validate_trigger_patches


def _camera():
    return (
        [
            {"pixel_id": 0, "type_id": 1, "x_cm": 0.0, "y_cm": 0.0},
            {"pixel_id": 1, "type_id": 1, "x_cm": 1.0, "y_cm": 0.0},
            {"pixel_id": 2, "type_id": 1, "x_cm": 5.0, "y_cm": 0.0},
        ],
        [{"type_id": 1, "funnel_diameter_cm": 1.0}],
    )


def _trigger(kind="majority"):
    return [{"group_id": 0, "kind": kind, "use_default_multiplicity": True}]


def test_validate_trigger_patches_accepts_connected_majority_patch():
    pixels, types = _camera()
    members = [
        {"group_id": 0, "member_order": 0, "pixel_order": 0, "pixel_id": 0, "required": 1},
        {"group_id": 0, "member_order": 0, "pixel_order": 1, "pixel_id": 1, "required": 0},
    ]

    assert validate_trigger_patches(pixels, types, _trigger(), members) == []


def test_validate_trigger_patches_reports_disconnected_patch():
    pixels, types = _camera()
    members = [
        {"group_id": 0, "member_order": 0, "pixel_order": 0, "pixel_id": 0},
        {"group_id": 0, "member_order": 0, "pixel_order": 1, "pixel_id": 2},
    ]

    diagnostics = validate_trigger_patches(
        pixels,
        types,
        _trigger(),
        members,
        raise_on_error=False,
    )
    assert {item["code"] for item in diagnostics} == {"disconnected_patch"}


@pytest.mark.parametrize(
    ("kind", "members", "message"),
    [
        (
            "analogsum",
            [
                {"group_id": 0, "member_order": 0, "pixel_order": 0, "pixel_id": 0},
                {"group_id": 0, "member_order": 0, "pixel_order": 1, "pixel_id": 1},
            ],
            "AnalogSumTrigger",
        ),
        (
            "digitalsum",
            [{"group_id": 0, "member_order": 0, "pixel_order": 0, "pixel_id": 0, "required": 1}],
            "required marker",
        ),
    ],
)
def test_validate_trigger_patches_rejects_invalid_simtel_syntax(kind, members, message):
    pixels, types = _camera()
    with pytest.raises(ValueError, match=message):
        validate_trigger_patches(pixels, types, _trigger(kind), members)


def test_validate_trigger_patches_rejects_duplicate_pixel():
    pixels, types = _camera()
    members = [
        {"group_id": 0, "member_order": 0, "pixel_order": 0, "pixel_id": 0},
        {"group_id": 0, "member_order": 0, "pixel_order": 1, "pixel_id": 0},
    ]
    with pytest.raises(ValueError, match="duplicate pixel"):
        validate_trigger_patches(pixels, types, _trigger(), members)
