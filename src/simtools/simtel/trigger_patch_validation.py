"""Stimulus planning and accounting for trigger-patch inspections."""

import hashlib
import itertools
import json
import math
import re
import shlex
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table

from simtools import __version__

INCONCLUSIVE = "inconclusive"
SECTORS = "sectors"
SECTOR_TYPE = "sector_type"
MC_EVENT = "mc_event"
SHOWER_NUMBER = "shower_num"
TELESCOPE_EVENTS = "telescope_events"
HEADER = "header"
LIST_TRGSECT = "list_trgsect"
TRIGGER_INFORMATION = "trigger_information"
TRIGGERED_TELESCOPES = "triggered_telescopes"


EVENT_ID = "event_id"
GROUP_ID = "group_id"
PLANNED = "planned"
OBSERVED = "observed"
INTENDED_SECTOR_FIRED = "intended_sector_fired"
MAPPING_STATUS = "mapping_status"
MISMATCH = "mismatch"
PIXEL_ID = "pixel_id"
FIRED_SECTORS = "fired_sectors"
NATIVE_PIXEL = "Pixel"
MODEL_MATCHES = "model_matches"
MODEL_GROUP_COUNT_MATCH = "model_group_count_match"
EVENTS_SUFFIX = ".events.ecsv"
ILLUMINATION_SUFFIX = ".illumination.dat"
HEADER_SECTOR_COUNT = "header_sector_count"
MEMBER_ORDER = "member_order"
PIXEL_ORDER = "pixel_order"
USE_DEFAULT_MULTIPLICITY = "use_default_multiplicity"

UTF_8 = "utf-8"
MASTERS = "masters"
REQUIRED = "required"
SLAVES = "slaves"
KIND = "kind"
MULTIPLICITY = "multiplicity"
EVENT_NUMBER = "event_number"
INPUT_PIXELS = "input_pixels"
SOURCE_GROUPS = "source_groups"
STATE = "state"
MISSING = "missing"
TRIGGERED = "triggered"
NON_TRIGGERED = "non_triggered"
UNRESOLVED = "unresolved"
PHOTOELECTRONS = "photoelectrons"
HEADER_MATCH = "header_match"
MODEL_MATCH = "model_match"
MEMBERS = "members"
KINDS = {"MajorityTrigger": 0, "AnalogSumTrigger": 1, "DigitalSumTrigger": 2}
NATIVE_MEMBER_PATTERN = re.compile(r"(\+?)(\d+)(?:\[(\d+(?:,\d+)*)\])?")


def read_camera(path):
    """Read native camera geometry and trigger roles without invoking the serializer."""
    groups, pixels = [], {}
    rotation = 0.0
    for line in Path(path).read_text(encoding=UTF_8).splitlines():
        tokens = shlex.split(line, comments=True)
        if not tokens:
            continue
        if tokens[0] == "Rotate":
            rotation = float(tokens[1])
        elif tokens[0] == NATIVE_PIXEL and len(tokens) >= 5:
            if len(tokens) < 10 or int(tokens[9]) != 0:
                pixels[int(tokens[1])] = (float(tokens[3]), float(tokens[4]))
        elif tokens[0] in KINDS:
            groups.append(_native_group(tokens))
    if not groups:
        raise ValueError(f"No trigger groups found in camera file: {path}")
    return groups, pixels, rotation


def _native_group(tokens):
    """Preserve mandatory masters and pre-sum slave roles in one native line."""
    if len(tokens) < 4 or tokens[2] != "of":
        raise ValueError(f"Malformed native trigger definition: {' '.join(tokens)}")
    masters, required, slaves = [], [], {}
    for token in tokens[3:]:
        member = NATIVE_MEMBER_PATTERN.fullmatch(token)
        if member is None:
            raise ValueError(f"Malformed native trigger member: {token}")
        master = int(member[2])
        masters.append(master)
        if member[1]:
            required.append(master)
        slaves[master] = [int(value) for value in member[3].split(",")] if member[3] else []
    return {
        KIND: KINDS[tokens[0]],
        MULTIPLICITY: tokens[1],
        MASTERS: masters,
        REQUIRED: required,
        SLAVES: slaves,
    }


def plan_patterns(
    groups,
    pixels,
    required_pixels,
    repetitions,
    include_slaves,
    exhaustive=False,
    max_events=None,
    max_radius=None,
):
    """Plan one deterministic combination per group, or explicit exhaustive combinations."""
    patterns = {}
    radius_cm = (max_radius * u.m).to_value(u.cm) if max_radius is not None else None
    pattern_limit = math.ceil(max_events / repetitions) if max_events is not None else None
    for group_id, group in enumerate(groups):
        combinations = _group_patterns(group, pixels, required_pixels, include_slaves, radius_cm)
        for pattern in combinations:
            if (
                pattern not in patterns
                and pattern_limit is not None
                and len(patterns) >= pattern_limit
            ):
                break
            patterns.setdefault(pattern, []).append(group_id)
            if not exhaustive:
                break
    return _pattern_rows(patterns, repetitions, max_events)


def _within_radius(pixel, pixels, radius_cm):
    """Select enabled pixels inside a radius in the native camera's centimetres."""
    return pixel in pixels and (radius_cm is None or math.hypot(*pixels[pixel]) <= radius_cm)


def _group_patterns(group, pixels, required_pixels, include_slaves, radius_cm):
    """Yield eligible patterns while preserving required masters and pre-sums."""
    eligible = [p for p in group[MASTERS] if _within_radius(p, pixels, radius_cm)]
    if required_pixels is None:
        required_pixels = len(group[MASTERS])
    required = group[REQUIRED]
    if not set(required) <= set(eligible) or len(eligible) < required_pixels:
        return
    optional = [pixel for pixel in eligible if pixel not in required]
    for combination in itertools.combinations(optional, max(0, required_pixels - len(required))):
        selected = set(required) | set(combination)
        if include_slaves:
            selected.update(slave for master in tuple(selected) for slave in group[SLAVES][master])
        if all(_within_radius(p, pixels, radius_cm) for p in selected):
            yield tuple(sorted(selected))


def _pattern_rows(patterns, repetitions, max_events):
    """Expand repeated trials into stable, one-based event identifiers."""
    rows = []
    for pattern, group_ids in patterns.items():
        for _ in range(repetitions):
            if max_events is not None and len(rows) >= max_events:
                return rows
            rows.append(
                {
                    EVENT_NUMBER: len(rows) + 1,
                    INPUT_PIXELS: json.dumps(pattern),
                    SOURCE_GROUPS: json.dumps(group_ids),
                }
            )
    if not rows:
        raise ValueError(
            "No illumination patterns selected; widen the radius or reduce required_pixels"
        )
    return rows


def write_stimulus_camera(source, destination, patterns, mirror_class):
    """Write exact patterns and compensate the documented single-mirror bypass transform."""
    _groups, _pixels, rotation = read_camera(source)
    cosine, sine = np.cos(np.deg2rad(rotation)), np.sin(np.deg2rad(rotation))
    lines = []
    for line in Path(source).read_text(encoding=UTF_8).splitlines():
        tokens = shlex.split(line, comments=True)
        if tokens and tokens[0] in KINDS:
            continue
        if tokens and tokens[0] == NATIVE_PIXEL and len(tokens) >= 5 and mirror_class in (0, 1):
            x, y = float(tokens[3]), float(tokens[4])
            # sim_telarray.c: single-mirror Bypass_Optics applies this involution.
            tokens[3] = str(x * cosine - y * sine)
            tokens[4] = str(-x * sine - y * cosine)
            line = " ".join(tokens)
        lines.append(line)
    for row in patterns:
        members = " ".join(f"+{pixel}" for pixel in json.loads(row[INPUT_PIXELS]))
        lines.append(f"MajorityTrigger * of {members}")
    Path(destination).write_text("\n".join(lines) + "\n", encoding=UTF_8)


def compare_model(groups, group_table, member_table):
    """Compare native roles against independent resolved model tables."""
    names = {"majority": 0, "analogsum": 1, "digitalsum": 2}
    matches = []
    for index, native in enumerate(groups):
        if index >= len(group_table):
            matches.append(False)
            continue
        expected = group_table[index]
        rows = [row for row in member_table if int(row[GROUP_ID]) == index]
        by_order = {}
        for row in rows:
            by_order.setdefault(int(row[MEMBER_ORDER]), []).append(row)
        roles = []
        for _order, members in sorted(by_order.items()):
            members.sort(key=lambda row: int(row[PIXEL_ORDER]))
            roles.append(
                (
                    int(members[0][PIXEL_ID]),
                    bool(members[0][REQUIRED]),
                    tuple(int(row[PIXEL_ID]) for row in members[1:]),
                )
            )
        actual = [
            (master, master in native[REQUIRED], tuple(native[SLAVES][master]))
            for master in native[MASTERS]
        ]
        multiplicity = "*" if expected[USE_DEFAULT_MULTIPLICITY] else str(expected[MULTIPLICITY])
        matches.append(
            names[str(expected[KIND]).lower()] == native[KIND]
            and multiplicity == native[MULTIPLICITY]
            and roles == actual
        )
    return matches, len(group_table) == len(groups)


def _header_memberships(organization):
    """Invert eventio's pixel-to-sector lists without assuming exclusive sectors."""
    memberships = [set() for _ in range(int(organization["n_sectors"]))]
    for pixel, sectors in enumerate(organization[SECTORS]):
        for sector in sectors:
            if 0 <= int(sector) < len(memberships):
                memberships[int(sector)].add(pixel)
    return memberships


def model_status(matches):
    """Keep unavailable source-model evidence distinct from a mismatch."""
    if matches is None:
        return "unavailable"
    return "match" if matches else MISMATCH


def member_pixels(group):
    """Flatten membership for comparisons that do not expose master/slave roles."""
    return set(group[MASTERS]) | {pixel for values in group[SLAVES].values() for pixel in values}


def build_group_rows(groups, organization, model_matches):
    """Compare each native group with simulator membership and trigger kind."""
    memberships = _header_memberships(organization)
    rows = []
    for group_id, group in enumerate(groups):
        members = member_pixels(group)
        present = group_id < len(memberships)
        matches = (
            present
            and members == memberships[group_id]
            and group[KIND] == int(organization[SECTOR_TYPE][group_id])
        )
        rows.append(
            {
                GROUP_ID: group_id,
                KIND: group[KIND],
                MEMBERS: json.dumps(sorted(members)),
                MODEL_MATCH: model_status(model_matches[group_id]),
                HEADER_MATCH: bool(matches),
                "header_threshold": float(organization["sector_threshold"][group_id])
                if present
                else float("nan"),
                "header_pixel_threshold": float(organization["sector_pixel_threshold"][group_id])
                if present
                else float("nan"),
                PLANNED: 0,
                OBSERVED: 0,
                TRIGGERED: 0,
                INTENDED_SECTOR_FIRED: 0,
            }
        )
    return rows


def account_events(events, telescope_id, patterns, group_rows, sample_limit=4):
    """Preserve every planned trial, including known non-triggers and missing records."""
    rows = [
        dict(
            row,
            state=MISSING,
            event_id=-1,
            fired_sectors="[]",
            pe_pixels="[]",
            mapping_status=INCONCLUSIVE,
        )
        for row in patterns
    ]
    seen, unexpected, duplicates, samples = set(), [], [], []
    for event in events:
        number = event.get(MC_EVENT, {}).get(SHOWER_NUMBER)
        if number is None or not 1 <= int(number) <= len(rows):
            unexpected.append(event.get(EVENT_ID, -1))
            continue
        index = int(number) - 1
        if index in seen:
            duplicates.append(int(number))
            continue
        seen.add(index)
        row = rows[index]
        _record_event(row, event, telescope_id)
        if len(samples) < sample_limit:
            samples.append(event)
    _group_trial_counts(rows, group_rows)
    return rows, unexpected, duplicates, samples


def _record_event(row, event, telescope_id):
    """Read the decision independently of whether telescope waveforms were retained."""
    telescope_event = event.get(TELESCOPE_EVENTS, {}).get(telescope_id, {})
    trigger_info = event.get(TRIGGER_INFORMATION, {})
    fired = telescope_event.get(HEADER, {}).get(LIST_TRGSECT, [])
    triggered_ids = trigger_info.get(TRIGGERED_TELESCOPES)
    if triggered_ids is not None:
        row[STATE] = TRIGGERED if telescope_id in triggered_ids else NON_TRIGGERED
    elif len(fired) > 0:
        row[STATE] = TRIGGERED
    elif telescope_event:
        row[STATE] = UNRESOLVED
    else:
        row[STATE] = NON_TRIGGERED
    row[EVENT_ID] = event[EVENT_ID]
    row[FIRED_SECTORS] = json.dumps([int(value) for value in fired])
    photoelectrons = event.get(PHOTOELECTRONS, {})
    pe_data = next(iter(photoelectrons.values()), {})
    pe_pixels = np.flatnonzero(pe_data.get(PHOTOELECTRONS, np.array([])) > 0).tolist()
    row["pe_pixels"] = json.dumps(pe_pixels)
    if pe_pixels:
        row[MAPPING_STATUS] = (
            "consistent" if set(pe_pixels) <= set(json.loads(row[INPUT_PIXELS])) else MISMATCH
        )


def _group_trial_counts(rows, group_rows):
    """Count coverage for every source group, including deduplicated patterns."""
    for row in rows:
        fired = json.loads(row[FIRED_SECTORS])
        for group_id in json.loads(row[SOURCE_GROUPS]):
            group = group_rows[group_id]
            group[PLANNED] += 1
            group[OBSERVED] += row[STATE] != MISSING
            group[TRIGGERED] += row[STATE] == TRIGGERED
            group[INTENDED_SECTOR_FIRED] += group_id in fired
    for group in group_rows:
        group["functional_status"] = INCONCLUSIVE if group[PLANNED] else "not_tested"


def save_results(pdf_file, rows, groups, summary, camera_file, metadata):
    """Write complete event/group tables and a compact provenance record."""
    path = Path(pdf_file)
    Table(rows=rows, meta=metadata).write(path.with_suffix(EVENTS_SUFFIX), overwrite=True)
    Table(
        rows=groups,
        meta={
            "threshold_convention": "Raw header values; normalized thresholds are not compared.",
            "kind_codes": "0=majority, 1=analog sum, 2=digital sum",
        },
    ).write(path.with_suffix(".groups.ecsv"), overwrite=True)
    summary["camera_sha256"] = file_digest(camera_file)
    summary["simtools_version"] = __version__
    summary["functional_acceptance"] = "inconclusive: photon operating point is not calibrated"
    summary["header_limits"] = "Simulator headers do not establish required-master/pre-sum roles."
    path.with_suffix(".summary.json").write_text(
        json.dumps(
            {key: value for key, value in summary.items() if key != MODEL_MATCHES},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding=UTF_8,
    )


def file_digest(path):
    """Hash a retained configuration artifact for provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
