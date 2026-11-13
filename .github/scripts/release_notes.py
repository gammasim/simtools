"""Render GitHub release notes from the changelog and published image tags."""

import argparse
import json
from pathlib import Path

IMAGE_NAME = "ghcr.io/gammasim/simtools-prod"


def _changelog_section(changelog, release_tag):
    """Return the complete changelog section for one release tag."""
    lines = changelog.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.startswith(f"## [{release_tag}]")),
        None,
    )
    if start is None:
        raise ValueError(f"No changelog section found for {release_tag}.")
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    return "\n".join(lines[start:end]).strip()


def render_release_notes(release_tag, changelog, template, image_tags):
    """Render release notes from the matching changelog section and image tags."""
    matching_tags = sorted({tag for tag in image_tags if tag.startswith(f"{release_tag}-")})
    if not matching_tags:
        raise ValueError(f"No production image tags found for {release_tag}.")
    images = "\n".join(f"- `{IMAGE_NAME}:{tag}`" for tag in matching_tags)
    return template.replace("{{ changelog }}", _changelog_section(changelog, release_tag)).replace(
        "{{ images }}", images
    )


def main():
    """Render a release note body to the requested output file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_tag")
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--template", type=Path, default=Path(".github/release-template.md"))
    parser.add_argument("--image-tags", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    image_tags = json.loads(arguments.image_tags.read_text(encoding="utf-8"))
    release_notes = render_release_notes(
        arguments.release_tag,
        arguments.changelog.read_text(encoding="utf-8"),
        arguments.template.read_text(encoding="utf-8"),
        image_tags,
    )
    arguments.output.write_text(release_notes + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
