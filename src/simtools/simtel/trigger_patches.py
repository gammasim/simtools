"""Validation helpers for camera trigger groups and pre-summed patches."""

from collections import defaultdict


def validate_trigger_patches(
    pixels,
    pixel_types,
    triggers,
    trigger_members,
    *,
    raise_on_error=True,
    spacing_tolerance_cm=0.2,
):
    """Validate normalized trigger groups and their pixel patches.

    A ``+`` marker is represented by ``required`` and is valid only for the
    first pixel of a majority-trigger member. Bracketed members are
    pre-summed inputs; their pixels must form a connected graph using the
    configured pixel diameters.

    Parameters
    ----------
    pixels : sequence of dict
        Camera pixels with ``pixel_id``, ``type_id``, ``x_cm``, and ``y_cm``.
    pixel_types : sequence of dict
        Camera pixel types with ``type_id`` and ``funnel_diameter_cm``.
    triggers : sequence of dict
        Trigger-group records.
    trigger_members : sequence of dict
        Normalized trigger-member records.
    raise_on_error : bool, optional
        Raise ``ValueError`` when an error is found. If false, return all
        diagnostics instead.
    spacing_tolerance_cm : float, optional
        Tolerance added to the sum of two pixel radii for adjacency. The
        default also accommodates the rounded coordinates in model tables.

    Returns
    -------
    list of dict
        Diagnostics with ``code``, ``group_id``, and ``message`` fields.

    Raises
    ------
    ValueError
        If ``raise_on_error`` is true and validation finds an error.
    """
    diagnostics = []
    pixel_map = {pixel.get("pixel_id"): pixel for pixel in pixels}
    type_map = {item.get("type_id"): item for item in pixel_types}
    group_ids = [item.get("group_id") for item in triggers]
    if group_ids != list(range(len(triggers))):
        diagnostics.append(
            _diagnostic(
                "group_ids", None, "Camera trigger group IDs must be contiguous and ordered"
            )
        )

    members_by_group = defaultdict(list)
    for member in trigger_members:
        members_by_group[member.get("group_id")].append(member)
    for group_id in set(members_by_group) - set(group_ids):
        diagnostics.append(
            _diagnostic(
                "unknown_group",
                group_id,
                "Camera trigger member references an unknown group",
            )
        )

    for trigger in triggers:
        group_id = trigger.get("group_id")
        rows = members_by_group.get(group_id, [])
        diagnostics.extend(
            _validate_group(trigger, rows, pixel_map, type_map, spacing_tolerance_cm)
        )

    if raise_on_error and diagnostics:
        raise ValueError("; ".join(item["message"] for item in diagnostics))
    return diagnostics


def _validate_group(trigger, rows, pixel_map, type_map, spacing_tolerance_cm):
    """Validate one trigger group."""
    diagnostics = []
    group_id = trigger.get("group_id")
    kind = str(trigger.get("kind", "")).lower()
    if kind not in {"majority", "analogsum", "digitalsum"}:
        diagnostics.append(
            _diagnostic("kind", group_id, f"Unsupported camera trigger kind: {kind}")
        )

    use_default = bool(trigger.get("use_default_multiplicity"))
    multiplicity = trigger.get("multiplicity")
    if use_default and multiplicity not in (None, 0):
        diagnostics.append(
            _diagnostic(
                "multiplicity",
                group_id,
                "Default trigger multiplicity must not be positive",
            )
        )
    elif not use_default and (multiplicity is None or int(multiplicity) < 1):
        diagnostics.append(
            _diagnostic("multiplicity", group_id, "Explicit trigger multiplicity must be positive")
        )
    if not rows:
        diagnostics.append(
            _diagnostic("empty_group", group_id, f"Camera trigger group has no members: {group_id}")
        )
        return diagnostics

    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("member_order")].append(row)
    member_orders = sorted(grouped)
    if member_orders != list(range(len(member_orders))):
        diagnostics.append(
            _diagnostic(
                "member_orders", group_id, "Camera trigger member orders must be contiguous"
            )
        )

    seen_pixels = set()
    for member_order, member_rows in grouped.items():
        ordered = sorted(member_rows, key=lambda row: row.get("pixel_order", -1))
        diagnostics.extend(
            _validate_member(
                ordered,
                group_id,
                member_order,
                kind,
                seen_pixels,
                pixel_map,
                type_map,
                spacing_tolerance_cm,
            )
        )
        seen_pixels.update(row.get("pixel_id") for row in ordered)
    return diagnostics


def _validate_member(
    rows,
    group_id,
    member_order,
    kind,
    seen_pixels,
    pixel_map,
    type_map,
    spacing_tolerance_cm,
):
    """Validate one normalized trigger member."""
    diagnostics = []
    pixel_orders = [row.get("pixel_order") for row in rows]
    if pixel_orders != list(range(len(rows))):
        diagnostics.append(
            _diagnostic("pixel_orders", group_id, "Camera trigger pixel orders must be contiguous")
        )
    pixel_ids = [row.get("pixel_id") for row in rows]
    if len(set(pixel_ids)) != len(pixel_ids) or any(
        pixel_id in seen_pixels for pixel_id in pixel_ids
    ):
        diagnostics.append(
            _diagnostic(
                "duplicate_pixel", group_id, "Camera trigger patch contains duplicate pixel IDs"
            )
        )
    if any(pixel_id not in pixel_map for pixel_id in pixel_ids):
        diagnostics.append(
            _diagnostic("unknown_pixel", group_id, "Camera trigger contains an unknown pixel ID")
        )
        return diagnostics
    required = [bool(row.get("required")) for row in rows]
    if any(required[1:]):
        diagnostics.append(
            _diagnostic(
                "required_order",
                group_id,
                "Only the first pixel of a trigger member may be required",
            )
        )
    if any(required) and kind != "majority":
        diagnostics.append(
            _diagnostic(
                "required_kind",
                group_id,
                "The '+' required marker is supported only for majority triggers",
            )
        )
    if len(rows) > 1 and kind == "analogsum":
        diagnostics.append(
            _diagnostic(
                "analogsum_patch",
                group_id,
                "AnalogSumTrigger does not support bracketed pre-summed members",
            )
        )
    diagnostics.extend(
        _validate_member_geometry(
            rows, group_id, member_order, pixel_map, type_map, spacing_tolerance_cm
        )
    )
    return diagnostics


def _validate_member_geometry(rows, group_id, member_order, pixel_map, type_map, tolerance):
    """Validate patch size and pixel adjacency."""
    diagnostics = []
    if len(rows) > 8:
        diagnostics.append(
            _diagnostic(
                "patch_size",
                group_id,
                "Camera trigger patch cannot contain more than 8 pixels",
            )
        )
    if len(rows) > 1 and not _is_connected(rows, pixel_map, type_map, tolerance):
        diagnostics.append(
            _diagnostic(
                "disconnected_patch",
                group_id,
                f"Camera trigger patch {member_order} contains non-neighbouring pixels",
            )
        )
    return diagnostics


def _is_connected(rows, pixel_map, type_map, spacing_tolerance_cm):
    """Return whether a patch's pixel-centre graph is connected."""
    positions = []
    for row in rows:
        pixel = pixel_map[row["pixel_id"]]
        pixel_type = type_map.get(pixel.get("type_id"), {})
        diameter = float(pixel_type.get("funnel_diameter_cm", 0.0))
        positions.append((float(pixel["x_cm"]), float(pixel["y_cm"]), diameter))
    connected = {0}
    while True:
        reached = {
            index
            for index in range(len(positions))
            if index not in connected
            and any(
                _are_neighbours(positions[index], positions[other], spacing_tolerance_cm)
                for other in connected
            )
        }
        if not reached:
            return len(connected) == len(positions)
        connected.update(reached)


def _are_neighbours(first, second, spacing_tolerance_cm):
    """Return whether two pixel centres are within one pixel diameter."""
    dx = first[0] - second[0]
    dy = first[1] - second[1]
    distance = (dx * dx + dy * dy) ** 0.5
    return distance <= (first[2] + second[2]) / 2 + spacing_tolerance_cm


def _diagnostic(code, group_id, message):
    """Build one validation diagnostic."""
    return {"code": code, "group_id": group_id, "message": message}
