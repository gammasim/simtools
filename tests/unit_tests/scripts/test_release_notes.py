import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / ".github/scripts/release_notes.py"
SPEC = importlib.util.spec_from_file_location("release_notes", SCRIPT_PATH)
RELEASE_NOTES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE_NOTES)


def test_release_notes_include_all_changelog_sections_and_matching_images(tmp_test_directory):
    changelog = Path(tmp_test_directory) / "CHANGELOG.md"
    template = Path(tmp_test_directory) / "template.md"
    changelog.write_text(
        "## [v1.2.3](url) - 2026-11-10\n\n### Features\n\nFeature text.\n\n"
        "### Fixes\n\nFix text.\n\n## [v1.2.2](url) - 2026-10-01\n\nOld release.\n",
        encoding="utf-8",
    )
    template.write_text("{{ changelog }}\n\n{{ images }}\n", encoding="utf-8")

    body = RELEASE_NOTES.render_release_notes(
        "v1.2.3",
        changelog.read_text(encoding="utf-8"),
        template.read_text(encoding="utf-8"),
        ["v0.1.2-v1-g1-generic", "v1.2.3-v1-g2-generic", "v1.2.3-v1-g2-avx2"],
    )

    assert "### Features\n\nFeature text." in body
    assert "### Fixes\n\nFix text." in body
    assert "Old release." not in body
    assert "ghcr.io/gammasim/simtools-prod:v1.2.3-v1-g2-generic" in body
    assert "ghcr.io/gammasim/simtools-prod:v1.2.3-v1-g2-avx2" in body
    assert "v0.1.2-v1-g1-generic" not in body


def test_release_notes_fail_when_changelog_section_or_image_missing():
    with pytest.raises(ValueError, match="No changelog section found"):
        RELEASE_NOTES.render_release_notes(
            "v9.9.9", "## [v1.0.0](url)\n", "{{ changelog }}", ["v9.9.9-v1-g2-generic"]
        )
    with pytest.raises(ValueError, match="No production image tags found"):
        RELEASE_NOTES.render_release_notes("v1.0.0", "## [v1.0.0](url)\n", "{{ changelog }}", [])


def test_release_notes_cli_writes_rendered_file(tmp_test_directory, monkeypatch):
    directory = Path(tmp_test_directory)
    changelog = directory / "changelog.md"
    template = directory / "template.md"
    image_tags = directory / "image-tags.json"
    output = directory / "release-notes.md"
    changelog.write_text("## [v1.2.3](url)\n\nRelease text.\n", encoding="utf-8")
    template.write_text("Changes:\n{{ changelog }}\nImages:\n{{ images }}", encoding="utf-8")
    image_tags.write_text('["v1.2.3-v1-g2-generic", "v1.2.2-old"]', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "v1.2.3",
            "--changelog",
            str(changelog),
            "--template",
            str(template),
            "--image-tags",
            str(image_tags),
            "--output",
            str(output),
        ],
    )

    RELEASE_NOTES.main()

    assert output.read_text(encoding="utf-8") == (
        "Changes:\n## [v1.2.3](url)\n\nRelease text.\nImages:\n"
        "- `ghcr.io/gammasim/simtools-prod:v1.2.3-v1-g2-generic`\n"
    )
