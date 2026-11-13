"""Resolve released setting-workflow runtimes to immutable references."""

import json
import re
from importlib.metadata import metadata
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

import yaml

from simtools.model_repository import dependency_versions


def _read_url(url):
    """Read a public release resource."""
    with urlopen(Request(url, headers={"User-Agent": "simtools"}), timeout=30) as response:
        return response.read()


def _simtools_repository():
    """Read the simtools source repository from the installed package metadata."""
    for entry in metadata("gammasimtools").get_all("Project-URL", []):
        label, _, url = entry.partition(",")
        if label.strip().lower() == "repository":
            repository = urlsplit(url.strip())
            if repository.scheme == "https" and repository.hostname == "github.com":
                return repository.path.strip("/").removesuffix(".git")
    raise ValueError("Installed simtools metadata does not specify a GitHub repository.")


def latest_models_commit(catalog):
    """Return the commit of the latest stable model release.

    Parameters
    ----------
    catalog : dict
        Dependency catalog containing the simulation-model repository URL.

    Returns
    -------
    str
        Full Git commit ID of the latest published stable release.
    """
    repository = urlsplit(catalog["model-repository"]["repository-url"])
    if repository.scheme != "https" or not repository.netloc:
        raise ValueError("The model repository URL must be an HTTPS GitLab URL.")
    project = quote(repository.path.strip("/").removesuffix(".git"), safe="")
    url = f"https://{repository.netloc}/api/v4/projects/{project}/releases?per_page=100"
    releases = json.loads(_read_url(url))
    for release in releases:
        if not release.get("upcoming_release") and re.fullmatch(
            r"v?\d+\.\d+\.\d+", release.get("tag_name", "")
        ):
            commit = release["commit"]["id"]
            if not re.fullmatch(r"[a-f0-9]{40}", commit):
                raise ValueError("The simulation-models release has an invalid commit ID.")
            return commit
    raise ValueError("No stable simulation-models release was found.")


def latest_simtools_image(catalog):
    """Resolve the latest simtools production image.

    Parameters
    ----------
    catalog : dict
        Dependency catalog containing the configured image registry.

    Returns
    -------
    str
        Multi-platform production image reference pinned by digest.
    """
    repository = _simtools_repository()
    release_url = f"https://api.github.com/repos/{repository}/releases/latest"
    release = json.loads(_read_url(release_url))["tag_name"]
    if not re.fullmatch(r"v\d+\.\d+\.\d+", release):
        raise ValueError(f"Unexpected simtools release tag: {release!r}")
    catalog_url = (
        f"https://raw.githubusercontent.com/{repository}/{release}/dependency_versions.yml"
    )
    release_catalog = yaml.safe_load(_read_url(catalog_url))
    matrix = dependency_versions.build_workflow_matrices(release_catalog)["production_matrix"]
    combination = next((item for item in matrix if item["avx_flag"] == "generic"), None)
    if combination is None:
        raise ValueError("The simtools release catalog does not define a generic production image.")
    tag = f"{release}-v{combination['corsika_build_id']}-{combination['simtel_build_id']}-generic"
    registry = catalog["images"]["registry"]
    host, _, namespace = registry.partition("/")
    if not host or not namespace:
        raise ValueError("The dependency catalog image registry must include a namespace.")
    image_path = f"{namespace}/simtools-prod"
    token_query = urlencode({"service": host, "scope": f"repository:{image_path}:pull"})
    token = json.loads(_read_url(f"https://{host}/token?{token_query}"))["token"]
    request = Request(
        f"https://{host}/v2/{image_path}/manifests/{tag}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.oci.image.index.v1+json, "
            "application/vnd.docker.distribution.manifest.list.v2+json",
            "User-Agent": "simtools",
        },
        method="HEAD",
    )
    with urlopen(request, timeout=30) as response:
        digest = response.headers.get("Docker-Content-Digest", "")
        media_type = response.headers.get("Content-Type", "").split(";", 1)[0]
    if media_type not in (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    ):
        raise ValueError(f"The simtools release image {tag} is not a multi-platform image.")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise ValueError(f"Registry did not return an image digest for {tag}.")
    return f"{registry}/simtools-prod@{digest}"
