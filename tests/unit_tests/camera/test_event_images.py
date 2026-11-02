"""Tests for event image preparation."""

from unittest import mock

import numpy as np
import pytest

from simtools.camera import event_images


def test_prepare_event_images_reads_events_and_camera():
    samples = np.ones((2, 3, 12))
    description = {
        "pixel_settings": {"time_slice": 1.0},
        "camera_settings": {"focal_length": 5.6},
    }
    with (
        mock.patch.object(
            event_images, "read_events", return_value=([4], description, [{"adc_samples": samples}])
        ),
        mock.patch.object(event_images, "Camera", return_value="camera") as camera,
    ):
        result = event_images.prepare_event_images("events.simtel", "LSTN-01", [4], 1)

    assert list(result.event_data) == [4]
    assert result.camera == "camera"
    assert result.time_axis.shape == (12,)
    camera.assert_called_once()


def test_prepare_event_images_rejects_empty_read():
    with mock.patch.object(event_images, "read_events", return_value=([], {}, [])):
        with pytest.raises(ValueError, match="No events read from file"):
            event_images.prepare_event_images("events.simtel", "LSTN-01")


def test_trace_maxima_selects_peaks_and_handles_empty_threshold():
    samples = np.array([[0, 1, 0, 0], [0, 0, 2, 0], [1, 1, 1, 1]])

    peaks, pixel_ids, found = event_images.trace_maxima(samples, sum_threshold=1)

    np.testing.assert_array_equal(pixel_ids, [1, 2])
    np.testing.assert_array_equal(peaks, [2, 0])
    assert found == 1
    assert event_images.trace_maxima(samples, sum_threshold=100) == (None, None, 0)
