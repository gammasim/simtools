"""Tests for trigger-patch plots."""

from simtools.visualization import plot_trigger_patches
from simtools.visualization.matplotlib_backend import pyplot as plt


def test_plot_trigger_patches_draws_patch_and_required_master():
    configuration = {
        "pixels": [
            {"pixel_id": 0, "x_cm": 0.0, "y_cm": 0.0},
            {"pixel_id": 1, "x_cm": 1.0, "y_cm": 0.0},
        ],
        "triggers": [{"group_id": 0, "kind": "majority"}],
        "trigger_members": [
            {"group_id": 0, "member_order": 0, "pixel_order": 0, "pixel_id": 0, "required": 1},
            {"group_id": 0, "member_order": 0, "pixel_order": 1, "pixel_id": 1, "required": 0},
        ],
    }

    figure = plot_trigger_patches.plot_trigger_patches(configuration, "Test")

    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) == 2
    assert figure.axes[0].get_title() == "Test trigger patches"
    plt.close(figure)
