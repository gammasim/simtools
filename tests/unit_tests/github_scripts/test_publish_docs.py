"""Tests for the GitHub Pages assembly script."""

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / ".github/scripts/publish_docs.py"
SPEC = importlib.util.spec_from_file_location("publish_docs", SCRIPT_PATH)
PUBLISH_DOCS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISH_DOCS)


def _entries(*versions):
    return [{"version": v, "url": f"https://example.org/{v}/"} for v in versions]


def _write_manifest(directory, entries):
    path = Path(directory) / "switcher.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _write_archive(path, files, extra=None):
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        for info in extra or []:
            archive.addfile(info)
    return path


def test_read_manifest_adds_names(tmp_test_directory):
    manifest = _write_manifest(tmp_test_directory, _entries("latest", "v1.2.3"))
    entries = PUBLISH_DOCS.read_manifest(manifest)
    assert [entry["name"] for entry in entries] == ["latest", "v1.2.3"]


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([], "nonempty list"),
        (["latest"], "must contain a version"),
        (_entries("v1.0.0"), "must include latest"),
        (_entries("latest", "1.0"), "Invalid switcher version"),
        (_entries("latest", "latest"), "Duplicate switcher version"),
        ([{"version": "latest"}], "https url"),
        ([{"version": "latest", "url": "http://example.org/"}], "https url"),
        ([{"version": "latest", "url": "https://x/", "name": "a b"}], "Invalid switcher name"),
    ],
)
def test_read_manifest_errors(tmp_test_directory, entries, message):
    manifest = _write_manifest(tmp_test_directory, entries)
    with pytest.raises(ValueError, match=message):
        PUBLISH_DOCS.read_manifest(manifest)


def test_extract_archive(tmp_test_directory):
    base = Path(tmp_test_directory)
    archive = _write_archive(base / "a.tar.gz", {"./index.html": b"abc", "sub/page.html": b"de"})
    size = PUBLISH_DOCS.extract_archive(archive, base / "out")
    assert size == 5
    assert (base / "out/sub/page.html").read_bytes() == b"de"


@pytest.mark.parametrize("name", ["../evil", "/abs", "c:/x", "a\\b"])
def test_extract_archive_rejects_unsafe_names(tmp_test_directory, name):
    base = Path(tmp_test_directory)
    archive = _write_archive(base / "a.tar.gz", {name: b"x"})
    with pytest.raises(ValueError, match=r"Unsafe archive member|Could not safely"):
        PUBLISH_DOCS.extract_archive(archive, base / "out")


def test_extract_archive_rejects_links_and_duplicates(tmp_test_directory):
    base = Path(tmp_test_directory)
    link = tarfile.TarInfo("link")
    link.type = tarfile.SYMTYPE
    link.linkname = "target"
    archive = _write_archive(base / "link.tar.gz", {}, extra=[link])
    with pytest.raises(ValueError, match="Unsafe archive member"):
        PUBLISH_DOCS.extract_archive(archive, base / "out1")
    archive = _write_archive(base / "dup.tar.gz", {"a": b"1"})
    with tarfile.open(archive, "w:gz") as handle:
        for _ in range(2):
            info = tarfile.TarInfo("a")
            info.size = 1
            handle.addfile(info, io.BytesIO(b"1"))
    with pytest.raises(ValueError, match="Duplicate archive member"):
        PUBLISH_DOCS.extract_archive(archive, base / "out2")


def test_extract_archive_size_limit(tmp_test_directory, monkeypatch):
    base = Path(tmp_test_directory)
    archive = _write_archive(base / "a.tar.gz", {"a": b"12345"})
    monkeypatch.setattr(PUBLISH_DOCS, "MAX_SITE_BYTES", 4)
    with pytest.raises(ValueError, match="exceeds the 1 GB"):
        PUBLISH_DOCS.extract_archive(archive, base / "out")


def test_extract_archive_not_an_archive(tmp_test_directory):
    base = Path(tmp_test_directory)
    (base / "bad.tar.gz").write_bytes(b"not a tarball")
    with pytest.raises(ValueError, match="Could not safely read archive"):
        PUBLISH_DOCS.extract_archive(base / "bad.tar.gz", base / "out")


def _prepare_site_inputs(base):
    manifest = _write_manifest(base, _entries("latest", "v1.0.0"))
    archives = base / "archives"
    archives.mkdir()
    _write_archive(archives / "v1.0.0.tar.gz", {"./index.html": b"v1"})
    latest = _write_archive(base / "latest.tar.gz", {"./index.html": b"latest"})
    return manifest, archives, latest


def test_assemble(tmp_test_directory):
    base = Path(tmp_test_directory)
    manifest, archives, latest = _prepare_site_inputs(base)
    site = base / "site"
    (site / "dev/test-resources").mkdir(parents=True)
    (site / "dev/test-resources/data.js").write_text("data", encoding="utf-8")
    (site / "stale.html").write_text("old", encoding="utf-8")

    PUBLISH_DOCS.assemble(site, manifest, archives, latest)

    assert (site / "index.html").read_bytes() == b"latest"
    assert (site / "v1.0.0/index.html").read_bytes() == b"v1"
    assert (site / "dev/test-resources/data.js").read_text(encoding="utf-8") == "data"
    assert (site / ".nojekyll").is_file()
    assert not (site / "stale.html").exists()
    switcher = json.loads((site / "_static/switcher.json").read_text(encoding="utf-8"))
    assert [entry["name"] for entry in switcher] == ["latest", "v1.0.0"]


def test_assemble_missing_inputs(tmp_test_directory):
    base = Path(tmp_test_directory)
    manifest, archives, latest = _prepare_site_inputs(base)
    (archives / "v1.0.0.tar.gz").unlink()
    with pytest.raises(ValueError, match="Missing archive"):
        PUBLISH_DOCS.assemble(base / "site", manifest, archives, latest)
    _write_archive(archives / "v1.0.0.tar.gz", {"index.html": b"v1"})
    with pytest.raises(ValueError, match="Latest documentation archive does not exist"):
        PUBLISH_DOCS.assemble(base / "site", manifest, archives, base / "none.tar.gz")


def test_assemble_requires_index(tmp_test_directory):
    base = Path(tmp_test_directory)
    manifest, archives, _ = _prepare_site_inputs(base)
    latest = _write_archive(base / "noindex.tar.gz", {"other.html": b"x"})
    with pytest.raises(ValueError, match=r"must contain index\.html"):
        PUBLISH_DOCS.assemble(base / "site", manifest, archives, latest)


def test_assemble_rejects_dashboard_symlink(tmp_test_directory):
    base = Path(tmp_test_directory)
    manifest, archives, latest = _prepare_site_inputs(base)
    site = base / "site"
    (site / "dev/test-resources").mkdir(parents=True)
    (site / "dev/test-resources/link").symlink_to(base)
    with pytest.raises(ValueError, match="symbolic link"):
        PUBLISH_DOCS.assemble(site, manifest, archives, latest)
