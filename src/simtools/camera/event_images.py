"""Prepare event images and camera data for visualization."""

from contextlib import closing
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from eventio import SimTelFile

from simtools.camera import trace_analysis as trace
from simtools.model.camera import Camera
from simtools.sim_events.simtel_io_metadata import (
    get_sim_telarray_telescope_id_to_telescope_name_mapping,
)
from simtools.simtel.simtel_event_reader import read_events

PEDESTAL_WINDOW_START = trace.PEDESTAL_WINDOW_START
INTEGRATION_WINDOW_START = trace.INTEGRATION_WINDOW_START
DEFAULT_SUM_THRESHOLD = trace.DEFAULT_SUM_THRESHOLD
EVENT_ID_KEY = "event_id"


@dataclass
class EventData:
    """Plain event arrays prepared for plotting."""

    event_id: int
    adc_samples: np.ndarray = None
    number_of_pixels: int = 0
    n_samples: int = 0
    pedestals: np.ndarray = None
    image: np.ndarray = None


@dataclass
class EventImages:
    """Prepared events, time axis, and camera pixel geometry."""

    event_data: dict
    time_axis: np.ndarray
    camera: object

    trace_maxima: object


def prepare_event_images(file_name, telescope, event_ids=None, max_events=None):
    """Read and prepare event arrays and camera geometry for plotting."""
    event_index, tel_desc, events = read_events(
        file_name, telescope, event_ids, max_events=max_events, verbose=True
    )
    if not events:
        raise ValueError(f"No events read from file {file_name}")

    event_data = {}
    n_samples = None
    for idx, event in zip(event_index, events):
        adc_samples, pedestals, image = trace.get_trace_data(event["adc_samples"])
        number_of_pixels, n_samples = adc_samples.shape
        event_data[idx] = EventData(
            event_id=idx,
            adc_samples=adc_samples,
            number_of_pixels=number_of_pixels,
            n_samples=n_samples,
            pedestals=pedestals,
            image=image,
        )

    time_axis = trace.get_time_axis(
        time_slice=tel_desc["pixel_settings"]["time_slice"] * u.ns, n_samples=n_samples
    )
    camera = Camera(
        telescope_name=telescope,
        camera_config_file=None,
        focal_length=tel_desc["camera_settings"]["focal_length"],
        camera_config_dict=tel_desc["camera_settings"],
    )
    return EventImages(
        event_data=event_data,
        time_axis=time_axis,
        camera=camera,
        trace_maxima=trace_maxima,
    )


def apply_monitoring_pedestals(event_images, file_name, telescope):
    """Replace waveform pedestals with selected events' monitoring values."""
    tel_map = get_sim_telarray_telescope_id_to_telescope_name_mapping(file_name)
    tel_id = next((key for key, name in tel_map.items() if name == telescope), None)
    if tel_id is None:
        raise ValueError(f"Telescope {telescope!r} is not present in the sim_telarray file")
    remaining = set(event_images.event_data)
    with closing(SimTelFile(file_name, skip_calibration=False)) as simtel_file:
        for event in simtel_file:
            event_id = event[EVENT_ID_KEY]
            if event_id not in remaining:
                continue
            data = event_images.event_data[event_id]
            monitoring = event.get("camera_monitorings", {}).get(tel_id)
            if monitoring is None:
                continue
            data.pedestals = _monitoring_pedestals(monitoring, data.number_of_pixels, event_id)
            data.image = trace.trace_integration(
                data.adc_samples, data.pedestals, (INTEGRATION_WINDOW_START, data.n_samples)
            )
            remaining.remove(event_id)
            if not remaining:
                return event_images
    if remaining:
        raise ValueError(f"No monitoring pedestal found for event IDs {sorted(remaining)}")
    return event_images


def trace_maxima(samples, sum_threshold):
    """Find per-pixel trace maxima using camera trace-analysis rules."""
    sums = samples.sum(axis=1)
    pixel_ids = np.flatnonzero(sums > float(sum_threshold))
    if pixel_ids.size == 0:
        return None, None, 0
    peak_samples = np.empty(pixel_ids.size, dtype=int)
    found_count = 0
    for index, pixel_id in enumerate(pixel_ids):
        pixel_trace = samples[pixel_id]
        peaks = trace.trace_maximum(pixel_trace)
        if peaks.size:
            found_count += 1
            peak_samples[index] = peaks[np.argmax(pixel_trace[peaks])]
        else:
            peak_samples[index] = np.argmax(pixel_trace)
    return peak_samples, pixel_ids, found_count


def _monitoring_pedestals(monitoring, number_of_pixels, event_id):
    """Extract calibrated per-sample high-gain pedestal values."""
    try:
        n_slices = monitoring["n_ped_slices"]
        values = np.asarray(monitoring["pedestal"], dtype=float)[0]
        if n_slices <= 0 or values.shape != (number_of_pixels,) or not np.all(np.isfinite(values)):
            raise ValueError
        return values / n_slices
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        raise ValueError(f"Missing or invalid monitoring pedestal for event {event_id}") from exc
