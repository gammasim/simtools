"""Characterize tabulated pulses and raw gain-resolved flasher waveforms."""

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table
from eventio import SimTelFile
from scipy.ndimage import gaussian_filter1d

from simtools.camera.waveform_metrics import measure_waveform
from simtools.simtel.simtel_io_metadata import (
    get_sim_telarray_telescope_id_to_telescope_name_mapping,
)
from simtools.visualization.plot_pulse_response import plot_response

LOW_GAIN_AMPLITUDE = "amplitude_low_gain"
PEAK_TIME = "peak_time_ns"
RESPONSE = "response"
PIXEL = "pixel"
INTEGRAL = "integral_adc_ns"
PEDESTAL = "pedestal"
TABLE_INTENSITY = "intensity"
WAVEFORM = "waveform"
FLASHER = "flasher"
EVENT_ID = "event_id"
TELESCOPE_EVENTS = "telescope_events"

FADC_PREFIX = "fadc_"
GAINS = ("HG", "LG")
INTENSITY = "illumination"
NONLINEAR = "nonlinear"
LINEAR = "linear_clipped"
AMPLITUDE = "amplitude"
TIME = "time"
ADC_SUM = "adc_sum"
PEAK = "peak_adc"
WIDTH = "width_ns"
WIDTH_FRACTION = "width_fraction"
THRESHOLD_ADC = "threshold_adc"


def template_responses(model, illuminations, width_fraction=0.95, threshold_adc=None):
    """Measure model pulse tables and a clipped-linear control at requested p.e. levels.

    Parameters
    ----------
    model : TelescopeModel
        Model supplying pulse tables and nominal ADC conversion parameters.
    illuminations : array-like
        Positive illumination levels in p.e.
    width_fraction : float
        Fraction of peak used for the pulse width.
    threshold_adc : float, optional
        Absolute width threshold overriding the fractional threshold.

    Returns
    -------
    astropy.table.Table, list
        Metrics and waveform dictionaries for both gains and response models.
        This is direct linear interpolation of the input tables, not a reproduction
        of the simulator's internal rpolator resampling or stochastic digitization.
    """
    levels = _positive_levels(illuminations)
    linear = model.get_parameter_table("fadc_pulse_shape")
    rows, waves = [], []
    for gain_index, gain in enumerate(GAINS):
        time, single, nonlinear, scale, pedestal, ceiling = _gain_response(
            model, linear, gain_index
        )
        dt = float(time[1] - time[0])
        for level in levels:
            pulse = _interpolate_response(nonlinear, time, single, level)
            for response, shape in ((NONLINEAR, pulse), (LINEAR, level * single)):
                wave = np.clip(shape * scale, -pedestal, ceiling)
                row = {
                    INTENSITY: level,
                    "gain": gain,
                    RESPONSE: response,
                    "above_table": level > np.max(nonlinear[TABLE_INTENSITY]),
                    **measure_waveform(wave, dt, width_fraction, threshold_adc),
                }
                row[PEAK_TIME] += time[0]
                rows.append(row)
                waves.append({**row, TIME: time, WAVEFORM: wave})
    table = _metrics_table(rows, "p.e.", width_fraction, threshold_adc)
    table.meta["interpolation"] = "Linear input-table interpolation; not simulator resampling"
    return table, waves


def _scalar(model, name):
    """Read a finite scalar model parameter, rejecting pixel-dependent arrays."""
    value = np.asarray(model.get_parameter_value(name), dtype=float).ravel()
    if value.size != 1 or not np.isfinite(value[0]):
        raise ValueError(f"Pulse response requires a finite scalar model parameter: {name}")
    return float(value[0])


def _time_ns(table):
    """Convert a model time column to ns, accepting bare ns values."""
    column = table[TIME]
    return np.asarray(u.Quantity(column, unit=column.unit or u.ns).to_value(u.ns))


def _gain_response(model, linear, gain_index):
    """Build one gain's uniformly sampled linear reference and ADC conversion."""
    gain = GAINS[gain_index].lower()
    nonlinear = model.get_parameter_table(f"{FADC_PREFIX}{gain}_pulse_shape")
    frequency = _scalar(model, "fadc_mhz")
    jitter = _scalar(model, "transit_time_jitter")
    if frequency <= 0 or jitter < 0:
        raise ValueError("Sampling frequency must be positive and jitter non-negative")
    dt = 1000 / frequency
    raw_time = _time_ns(linear)
    time_nl = _time_ns(nonlinear)
    start = min(raw_time.min(), time_nl.min()) - 5 * jitter
    stop = max(raw_time.max(), time_nl.max()) + 5 * jitter
    time = np.arange(start, stop + 0.5 * dt, dt)
    fine_dt = min(dt, 0.1)
    fine_time = np.arange(start, stop + fine_dt, fine_dt)
    column = AMPLITUDE
    if gain_index and LOW_GAIN_AMPLITUDE in linear.colnames:
        column = LOW_GAIN_AMPLITUDE
    raw = np.asarray(linear[column], dtype=float).copy()
    raw[raw_time < 0] = 0
    if not np.all(np.isfinite(raw)) or raw.max() <= 0:
        raise ValueError("Linear pulse must have a finite positive peak")
    fine_pulse = np.interp(fine_time, raw_time, raw / raw.max(), left=0, right=0)
    if jitter > 0:
        fine_pulse = gaussian_filter1d(fine_pulse, jitter / fine_dt, mode="constant")
    single = np.interp(time, fine_time, fine_pulse)
    scale, pedestal, ceiling = _gain_parameters(model, gain_index)
    return time, single, nonlinear, scale, pedestal, ceiling


def _gain_parameters(model, gain_index):
    """Resolve sim_telarray's negative LG defaults from the HG settings."""
    prefix = "fadc_lg_" if gain_index else FADC_PREFIX
    sensitivity = _scalar(model, f"{prefix}sensitivity")
    pedestal = _scalar(model, f"{prefix}{PEDESTAL}")
    if gain_index and sensitivity < 0:
        sensitivity = _scalar(model, "fadc_sensitivity")
    if gain_index and pedestal < 0:
        pedestal = _scalar(model, "fadc_pedestal")
    scale = _scalar(model, f"{prefix}{AMPLITUDE}") * sensitivity
    ceiling = _scalar(model, "fadc_max_signal") - pedestal
    if scale <= 0 or ceiling <= 0:
        raise ValueError("Gain and ADC headroom must be positive")
    return scale, pedestal, ceiling


def _interpolate_response(table, time, single, level):
    """Interpolate complete pulse amplitudes without applying intensity twice."""
    levels = np.unique(np.asarray(table[TABLE_INTENSITY], dtype=float))
    if levels.size < 2 or not np.all(np.isfinite(levels)) or levels[0] <= 0:
        raise ValueError("Non-linear pulse table requires two positive finite intensity levels")
    if level <= levels[0]:
        return single * level
    upper = min(np.searchsorted(levels, level), len(levels) - 1)
    lower = max(0, upper - 1)
    fraction = np.clip((level - levels[lower]) / (levels[upper] - levels[lower]), 0, 1)
    pulses = []
    for index in (lower, upper):
        selected = table[np.asarray(table[TABLE_INTENSITY]) == levels[index]]
        times = _time_ns(selected)
        amplitudes = np.asarray(selected[AMPLITUDE], dtype=float)
        if not np.all(np.diff(times) > 0) or not np.all(np.isfinite(amplitudes)):
            raise ValueError("Pulse table needs increasing times and finite amplitudes")
        pulses.append(np.interp(time, times, amplitudes, left=0, right=0))
    return (1 - fraction) * pulses[0] + fraction * pulses[1]


def _positive_levels(illuminations):
    """Validate scan coordinates."""
    levels = np.asarray(illuminations, dtype=float)
    if levels.ndim != 1 or not levels.size or not np.all(np.isfinite(levels) & (levels > 0)):
        raise ValueError("Provide positive finite illumination levels")
    return levels


def flasher_responses(
    files,
    illuminations,
    events_per_level,
    telescope,
    pixels,
    width_fraction=0.95,
    threshold_adc=None,
    pedestal=None,
):
    """Measure raw HG/LG flasher traces, using requested photons as the scan axis.

    Parameters
    ----------
    files : list of path-like
        One file per intensity, or one file containing the entire ordered scan.
    illuminations : list of float
        Requested photon counts, one per scan step, not detected p.e.
    events_per_level : list of int
        Expected event counts per intensity; one count broadcasts to all steps.
    telescope : str
        Telescope name stored in the file metadata.
    pixels : list of int
        Fixed pixel IDs to measure independently.
    width_fraction : float
        Fraction of peak for the width.
    threshold_adc : float, optional
        Absolute width threshold above pedestal.
    pedestal : float, optional
        Explicit per-sample pedestal for both gains. Otherwise use gain-specific
        monitoring pedestals; missing monitoring is an error.

    Returns
    -------
    astropy.table.Table, list
        Per-event metrics and mean-waveform dictionaries for each step, pixel and gain.
        All events must match the declared ordered scan; missing events are errors.
    """
    levels = _positive_levels(illuminations)
    counts = np.asarray(events_per_level, dtype=int)
    if counts.size == 1:
        counts = np.repeat(counts, len(levels))
    if counts.shape != levels.shape or np.any(counts <= 0):
        raise ValueError("Provide positive event counts, one or one per illumination step")
    if len(files) not in (1, len(levels)):
        raise ValueError("Provide one file for all steps or one file per illumination")
    if not pixels or len(set(pixels)) != len(pixels) or min(pixels) < 0:
        raise ValueError("Provide distinct non-negative pixel IDs")
    rows, sums = [], {}
    width_options = {WIDTH_FRACTION: width_fraction, THRESHOLD_ADC: threshold_adc}
    for index, file in enumerate(files):
        indices = np.arange(len(levels)) if len(files) == 1 else np.array([index])
        expected_steps = np.repeat(indices, counts[indices])
        for row, wave, dt in _read_scan_file(
            file, telescope, pixels, levels, expected_steps, width_options, pedestal
        ):
            rows.append(row)
            key = (row["step"], row[PIXEL], row["gain"])
            _accumulate_waveform(sums, key, wave, dt)
    waves = []
    for (step, pixel, gain), (total, count, dt) in sums.items():
        waves.append(
            {
                "step": step,
                PIXEL: pixel,
                "gain": gain,
                INTENSITY: levels[step],
                RESPONSE: FLASHER,
                TIME: np.arange(total.size) * dt,
                WAVEFORM: total / count,
            }
        )
    return _metrics_table(rows, "photons", width_fraction, threshold_adc), waves


def _accumulate_waveform(sums, key, wave, dt):
    """Accumulate equally sampled waveforms for one step, pixel and gain."""
    total, count, interval = sums.get(key, (np.zeros_like(wave), 0, dt))
    if total.shape != wave.shape or interval != dt:
        raise ValueError("Inconsistent waveform sampling within one scan step")
    sums[key] = (total + wave, count + 1, dt)


def _read_scan_file(file, telescope, pixels, levels, expected_steps, width_options, pedestal):
    """Yield measured gain channels from an ordered scan, checking event counts."""
    mapping = get_sim_telarray_telescope_id_to_telescope_name_mapping(file)
    tel_id = next((key for key, value in mapping.items() if value == telescope), None)
    if tel_id is None:
        raise ValueError(f"Telescope {telescope} is missing from {file}")
    count = 0
    with SimTelFile(file, skip_calibration=False) as stream:
        dt = stream.telescope_descriptions[tel_id]["pixel_settings"]["time_slice"]
        for event in stream:
            if event[TELESCOPE_EVENTS].get(tel_id) is None:
                raise ValueError(f"Missing telescope event in {file}; scan assignment is ambiguous")
            if count >= len(expected_steps):
                raise ValueError(f"Event count exceeds the declared scan in {file}")
            step = int(expected_steps[count])
            for pixel, gain, wave in _event_waveforms(event, tel_id, pixels, pedestal):
                row = {
                    "file": str(file),
                    EVENT_ID: event[EVENT_ID],
                    "step": step,
                    PIXEL: pixel,
                    "gain": gain,
                    RESPONSE: FLASHER,
                    INTENSITY: levels[step],
                    **measure_waveform(wave, dt, **width_options),
                }
                yield row, wave, dt
            count += 1
    if count != len(expected_steps):
        raise ValueError(f"Event count in {file}: expected {len(expected_steps)}, found {count}")


def _event_waveforms(event, tel_id, pixels, pedestal):
    """Yield pedestal-subtracted raw gain channels for the selected pixels."""
    adc = np.asarray(event[TELESCOPE_EVENTS][tel_id].get("adc_samples"), dtype=float)
    if adc.ndim != 3 or adc.shape[0] != 2 or max(pixels) >= adc.shape[1]:
        raise ValueError("Expected two raw gain channels and valid selected pixels")
    baselines = _pedestals(event, tel_id, adc.shape[:2], pedestal)
    for pixel in pixels:
        for gain_index, gain in enumerate(GAINS):
            yield pixel, gain, adc[gain_index, pixel] - baselines[gain_index, pixel]


def _pedestals(event, tel_id, shape, pedestal):
    """Convert integrated monitoring pedestals into per-sample HG/LG baselines."""
    if pedestal is not None:
        if not np.isfinite(pedestal):
            raise ValueError("Explicit pedestal must be finite")
        return np.full(shape, pedestal)
    try:
        monitoring = event["camera_monitorings"][tel_id]
        slices = monitoring["n_ped_slices"]
        values = np.asarray(monitoring[PEDESTAL], dtype=float)
        if slices <= 0 or values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError
        return values / slices
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing valid HG/LG monitoring pedestals; provide --pedestal") from exc


def _metrics_table(rows, axis, fraction, threshold):
    """Attach units and width conventions to response measurements."""
    table = Table(rows=rows)
    for name in (PEAK, ADC_SUM):
        table[name].unit = u.ct
    table[INTEGRAL].unit = u.ct * u.ns
    for name in (WIDTH, PEAK_TIME):
        table[name].unit = u.ns
    table.meta["illumination_unit"] = axis
    table.meta[WIDTH_FRACTION] = fraction
    table.meta[THRESHOLD_ADC] = threshold
    return table


def write_response(table, waves, output_directory, name):
    """Write response measurements, aggregate statistics, and diagnostic plots.

    Parameters
    ----------
    table : astropy.table.Table
        Per-event or per-template response measurements.
    waves : list of dict
        Waveforms with gain, illumination, response, time and waveform entries.
    output_directory : path-like
        Directory receiving ECSV tables and PDF figures.
    name : str
        Output filename stem.

    Returns
    -------
    pathlib.Path
        Path to the per-event or per-template ECSV table.
    """
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.ecsv"
    table.write(path, overwrite=True)
    summary = summarize_response(table)
    summary.write(directory / f"{name}_summary.ecsv", overwrite=True)
    plot_response(summary, waves, directory / f"{name}.pdf")
    return path


def summarize_response(table):
    """Aggregate event metrics per intensity, gain and pixel, retaining dispersion.

    Parameters
    ----------
    table : astropy.table.Table
        Response measurements.

    Returns
    -------
    astropy.table.Table
        Mean metrics, population standard deviations, counts and truncation counts.
    """
    keys = [INTENSITY, "gain", RESPONSE]
    if PIXEL in table.colnames:
        keys.append(PIXEL)
    rows = []
    for group in table.group_by(keys).groups:
        row = {key: group[key][0] for key in keys}
        row["event_count"] = len(group)
        row["truncated_count"] = int(np.sum(group["truncated"]))
        for name in (PEAK, ADC_SUM, INTEGRAL, WIDTH):
            values = np.asarray(group[name], dtype=float)
            finite = values[np.isfinite(values)]
            row[name] = float(np.mean(finite)) if finite.size else float("nan")
            row[f"{name}_std"] = float(np.std(finite)) if finite.size else float("nan")
        rows.append(row)
    summary = Table(rows=rows, meta=table.meta)
    for name in (PEAK, ADC_SUM, INTEGRAL, WIDTH):
        summary[name].unit = table[name].unit
        summary[f"{name}_std"].unit = table[name].unit
    return summary
