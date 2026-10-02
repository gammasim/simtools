#!/usr/bin/env python3
"""Apply resolved build-source revisions to the dependency catalog."""

import argparse
import json
from pathlib import Path

SOURCE_URL_KEYS = {
    "source-url: ",
    "config-source-url: ",
    "opt-patch-source-url: ",
    "hessio-source-url: ",
    "stdtools-source-url: ",
}


def _split_at_section(lines, section):
    """Return lines before and starting at a top-level catalog section."""
    for index, line in enumerate(lines):
        if line.startswith(f"{section}:"):
            return lines[: index + 1], lines[index + 1 :]
    raise ValueError(f"Missing dependency catalog section: {section}")


def _revision_key(source_url_key, section):
    """Return the revision field stored below a source URL field."""
    if source_url_key == "source-url: ":
        return "source-revision" if section == "corsika" else "revision"
    return source_url_key.removesuffix("source-url: ") + "revision"


def _replace_or_insert_revision(lines, index, revision_key, value):
    """Return the revision line and count of existing lines to skip."""
    revision_line = f"    {revision_key}: {value}\n"
    next_line = lines[index + 1] if index + 1 < len(lines) else ""
    skip_existing = int(next_line.startswith(f"    {revision_key}: "))
    return revision_line, skip_existing


def _source_url_key(stripped):
    """Return the matching source URL key, if the line declares one."""
    for key in SOURCE_URL_KEYS:
        if stripped.startswith(key):
            return key
    return None


def _is_next_section(line):
    """Return whether a line starts the next top-level YAML section."""
    return bool(line) and not line.startswith(" ")


def _revision_update(lines, index, current_ref, updates, section):
    """Return a revision line and existing-line count for the current URL line."""
    source_url_key = _source_url_key(lines[index].strip())
    component_updates = updates.get(current_ref, {})
    if source_url_key is None:
        return "", 0
    revision_key = _revision_key(source_url_key, section)
    value = component_updates.get(revision_key)
    if value is None:
        return "", 0
    return _replace_or_insert_revision(lines, index, revision_key, value)


def _update_component(lines, updates, section):
    """Insert or replace revision keys for one dependency-catalog section."""
    updated, remaining = _split_at_section(lines, section)
    current_ref = None
    index = 0
    while index < len(remaining):
        line = remaining[index]
        if _is_next_section(line):
            return updated, remaining[index:]
        updated.append(line)
        stripped = line.strip()
        if stripped.startswith("- source-ref: "):
            current_ref = stripped.removeprefix("- source-ref: ")
        revision_line, skipped = _revision_update(remaining, index, current_ref, updates, section)
        if revision_line:
            updated.append(revision_line)
            index += skipped
        index += 1
    return updated, []


def update_catalog(catalog_path, updates):
    """Write resolved source revisions while preserving catalog layout."""
    lines = catalog_path.read_text(encoding="utf-8").splitlines(keepends=True)
    before_simtel, simtel_and_after = _update_component(lines, updates["corsika"], "corsika")
    simtel, after = _update_component(simtel_and_after, updates["sim-telarray"], "sim-telarray")
    catalog_path.write_text("".join([*before_simtel, *simtel, *after]), encoding="utf-8")


def main():
    """Parse arguments and update the catalog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("updates", type=Path)
    arguments = parser.parse_args()
    updates = json.loads(arguments.updates.read_text(encoding="utf-8"))
    update_catalog(arguments.catalog, updates)


if __name__ == "__main__":
    main()
