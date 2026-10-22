import io
import json
import tarfile
from pathlib import Path

import pytest

from simtools.reporting import docs_pages

HOST = "https://docs.example.org"
INDEX_HTML = "index.html"
SHARED_CONTENT = "<html>shared</html>"


def _write_switcher(path, entries):
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _switcher_entries(*versions):
    return [
        {"version": "latest", "url": f"{HOST}/", "preferred": True},
        *({"version": version, "url": f"{HOST}/{version}/"} for version in versions),
    ]


def _write_site_tarball(registry, version, files):
    """Write a documentation tarball for a version to a local registry directory."""
    target = registry / version / "docs-site.tar.gz"
    target.parent.mkdir(parents=True)
    with tarfile.open(target, "w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_version_key():
    assert docs_pages.version_key("v0.10.0") > docs_pages.version_key("v0.9.1")


def test_read_version_switcher(tmp_test_directory):
    switcher = _write_switcher(
        Path(tmp_test_directory) / "switcher.json", _switcher_entries("v0.9.0", "v0.10.0")
    )
    assert docs_pages.read_version_switcher(switcher) == ["v0.10.0", "v0.9.0"]


def test_read_version_switcher_site_with_base_path(tmp_test_directory):
    entries = [
        {"version": "latest", "url": f"{HOST}/project/", "preferred": True},
        {"version": "v1.0.0", "url": f"{HOST}/project/v1.0.0/"},
    ]
    switcher = _write_switcher(Path(tmp_test_directory) / "switcher.json", entries)
    assert docs_pages.read_version_switcher(switcher) == ["v1.0.0"]


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([], "expected a non-empty list"),
        ({"version": "latest"}, "expected a non-empty list"),
        (
            [{"version": "v1.0.0", "url": f"{HOST}/v1.0.0/", "preferred": True}],
            "Exactly one 'latest' entry",
        ),
        (
            [*_switcher_entries("v1.0.0"), {"version": "latest", "url": f"{HOST}/"}],
            "Exactly one 'latest' entry",
        ),
        (
            [{"version": "latest", "url": "http://docs.example.org", "preferred": True}],
            "must start with 'https://' and end with '/'",
        ),
        ([*_switcher_entries(), {"version": "1.0.0", "url": f"{HOST}/1.0.0/"}], "match vX.Y.Z"),
        (
            [*_switcher_entries(), {"version": "v1.0.1", "url": f"{HOST}/v1.0.0/"}],
            "URL of v1.0.1 must be",
        ),
        (
            [{**entry, "preferred": False} for entry in _switcher_entries("v1.0.0")],
            "Exactly one entry must be marked",
        ),
        (_switcher_entries("v1.0.0", "v1.0.0"), "Duplicate versions"),
        (_switcher_entries(), "No released version"),
    ],
)
def test_read_version_switcher_invalid(tmp_test_directory, entries, message):
    switcher = _write_switcher(Path(tmp_test_directory) / "switcher.json", entries)
    with pytest.raises(ValueError, match=message):
        docs_pages.read_version_switcher(switcher)


def test_download_version(tmp_test_directory):
    tmp_path = Path(tmp_test_directory)
    _write_site_tarball(tmp_path / "registry", "v1.0.0", {INDEX_HTML: SHARED_CONTENT})

    docs_pages.download_version(
        "v1.0.0", f"file://{tmp_path}/registry/{{version}}/docs-site.tar.gz", tmp_path / "site"
    )

    assert (tmp_path / "site" / "v1.0.0" / INDEX_HTML).read_text() == SHARED_CONTENT


def test_download_version_sends_job_token(tmp_test_directory, mocker):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz"):
        pass
    response = mocker.MagicMock()
    response.__enter__.return_value.read.return_value = buffer.getvalue()
    urlopen = mocker.patch("urllib.request.urlopen", return_value=response)

    docs_pages.download_version(
        "v1.0.0", "https://registry/{version}/site.tar.gz", Path(tmp_test_directory), "secret"
    )

    request = urlopen.call_args[0][0]
    assert request.full_url == "https://registry/v1.0.0/site.tar.gz"
    assert request.get_header("Job-token") == "secret"


def test_copy_version_to_root(tmp_test_directory):
    site = Path(tmp_test_directory)
    (site / "v1.0.0" / "sub").mkdir(parents=True)
    (site / "v1.0.0" / INDEX_HTML).write_text("root")
    (site / "v1.0.0" / "sub" / "page.html").write_text("page")

    docs_pages.copy_version_to_root(site, "v1.0.0")

    assert (site / INDEX_HTML).read_text() == "root"
    assert (site / "sub" / "page.html").read_text() == "page"


def test_remove_source_maps(tmp_test_directory):
    site = Path(tmp_test_directory)
    (site / "theme.js").write_text("js")
    (site / "theme.js.map").write_text("map")

    docs_pages.remove_source_maps(site)

    assert (site / "theme.js").exists()
    assert not (site / "theme.js.map").exists()


def test_deduplicate(tmp_test_directory):
    site = Path(tmp_test_directory)
    (site / "v1.0.0" / "sub").mkdir(parents=True)
    (site / INDEX_HTML).write_text(SHARED_CONTENT)
    (site / "v1.0.0" / INDEX_HTML).write_text(SHARED_CONTENT)
    (site / "v1.0.0" / "sub" / "copy.html").write_text(SHARED_CONTENT)
    (site / "v1.0.0" / "unique.html").write_text("<html>unique</html>")

    docs_pages.deduplicate(site)

    assert not (site / INDEX_HTML).is_symlink()
    assert (site / "v1.0.0" / INDEX_HTML).is_symlink()
    assert (site / "v1.0.0" / INDEX_HTML).readlink() == Path("..") / INDEX_HTML
    assert (site / "v1.0.0" / "sub" / "copy.html").read_text() == SHARED_CONTENT
    assert not (site / "v1.0.0" / "unique.html").is_symlink()


def test_deployed_size_counts_symlinks_by_target_length(tmp_test_directory):
    site = Path(tmp_test_directory)
    (site / "file.html").write_text("x" * 100)
    (site / "link.html").symlink_to("file.html")

    assert docs_pages.deployed_size(site) == 100 + len("file.html")


def test_check_size(tmp_test_directory):
    site = Path(tmp_test_directory)
    (site / "v1.0.0").mkdir()
    (site / "v1.0.0" / INDEX_HTML).write_text("x" * 2000)

    docs_pages.check_size(site, ["v1.0.0"], size_budget_mb=0.01)
    with pytest.raises(RuntimeError, match="exceeds the budget"):
        docs_pages.check_size(site, ["v1.0.0"], size_budget_mb=0.001)


def test_assemble_pages(tmp_test_directory):
    tmp_path = Path(tmp_test_directory)
    registry = tmp_path / "registry"
    _write_site_tarball(
        registry, "v1.0.0", {INDEX_HTML: SHARED_CONTENT, "old.html": "old", "a.js.map": "map"}
    )
    _write_site_tarball(registry, "v1.1.0", {INDEX_HTML: SHARED_CONTENT, "new.html": "new"})
    switcher = _write_switcher(tmp_path / "switcher.json", _switcher_entries("v1.0.0", "v1.1.0"))
    site = tmp_path / "public"

    docs_pages.assemble_pages(
        switcher, site, f"file://{registry}/{{version}}/docs-site.tar.gz", size_budget_mb=1.0
    )

    assert (site / "new.html").read_text() == "new"
    assert not (site / "old.html").exists()
    assert (site / "v1.0.0" / "old.html").read_text() == "old"
    assert (site / "v1.1.0" / INDEX_HTML).is_symlink()
    assert (site / "v1.0.0" / INDEX_HTML).is_symlink()
    assert not list(site.rglob("*.map"))
    assert json.loads((site / docs_pages.SWITCHER_FILE_PATH).read_text()) == json.loads(
        switcher.read_text()
    )


def test_assemble_pages_requires_empty_output(tmp_test_directory):
    tmp_path = Path(tmp_test_directory)
    switcher = _write_switcher(tmp_path / "switcher.json", _switcher_entries("v1.0.0"))
    (tmp_path / "public").mkdir()
    (tmp_path / "public" / "file").write_text("x")

    with pytest.raises(FileExistsError, match="is not empty"):
        docs_pages.assemble_pages(switcher, tmp_path / "public", "file:///{version}", 1.0)
