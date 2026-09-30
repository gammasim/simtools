#!/usr/bin/python3
"""Validate cross-file invariants in a simtools data repository."""

import hashlib
from pathlib import Path

import pygit2
import yaml

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.io import ascii_handler

_DEFAULT_METADATA_ROOTS = (Path("input"), Path("output"))
_SKIP_MARKER = "SKIP_WORKFLOW_CI"
_SUBMIT_APPLICATION = "simtools-submit-model-parameter-from-external"
_HASH_CHUNK_SIZE = 1024 * 1024


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        cli.ArgumentDefinition(
            "repository",
            type=Path,
            default=Path(),
            help="Root directory of the data repository.",
        ),
        cli.ArgumentDefinition(
            "metadata_roots",
            type=Path,
            nargs="+",
            default=_DEFAULT_METADATA_ROOTS,
            help="Directories containing metadata and referenced products.",
        ),
        cli.ArgumentDefinition(
            "workflow_root",
            type=Path,
            default=Path("input"),
            help="Directory containing workflow configuration files.",
        ),
        cli.ArgumentDefinition(
            "require_git_tracking",
            action="store_true",
            help="Require metadata and product files to be tracked by Git.",
        ),
    ),
    setup_io_handler=False,
    resolve_sim_software_executables=False,
    initialize_model_reader=False,
)


def _tracked_files(repository):
    """Return paths tracked by the Git index."""
    try:
        git_repository = pygit2.Repository(str(repository))
    except (KeyError, OSError, ValueError, pygit2.GitError) as exc:
        raise ValueError(f"Not a readable Git repository: {repository}") from exc
    return {Path(entry.path) for entry in git_repository.index}


def _file_digest(file_path):
    """Return a SHA-256 digest without reading the entire file into memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        while chunk := file_handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_product(metadata_file, relative_metadata):
    """Read the product section from one metadata file."""
    try:
        product = ascii_handler.collect_data_from_file(metadata_file)["cta"]["product"]
        filename = product["filename"]
    except (OSError, TypeError, KeyError, ValueError, yaml.YAMLError) as exc:
        return None, None, (f"{relative_metadata}: cannot read cta.product.filename: {exc}")
    return product, filename, None


def _product_file(
    metadata_file,
    product,
    filename,
    repository,
    tracked,
    relative_metadata,
):
    """Return the valid product file named by a metadata document."""
    if filename is None:
        return None, None
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        return None, (f"{relative_metadata}: filename must name a file beside the metadata")

    product_files = [metadata_file.parent / filename]
    product_data = product.get("data", {})
    model = product_data.get("model", {}) if isinstance(product_data, dict) else {}
    model_name = model.get("name") if isinstance(model, dict) else None
    if (
        isinstance(model_name, str)
        and model_name not in {"", ".", ".."}
        and Path(model_name).name == model_name
    ):
        product_files.append(metadata_file.parent / model_name / filename)

    for product_file in product_files:
        relative_product = product_file.relative_to(repository)
        if product_file.is_file() and (tracked is None or relative_product in tracked):
            return product_file, None

    relative_product = product_files[0].relative_to(repository)
    return None, (f"{relative_metadata}: product file is missing or untracked: {relative_product}")


def _validate_metadata_file(metadata_file, repository, tracked):
    """Validate one metadata file and return its product identifier record."""
    relative_metadata = metadata_file.relative_to(repository)
    errors = []
    if tracked is not None and relative_metadata not in tracked:
        errors.append(f"{relative_metadata}: metadata file is untracked")
    product, filename, error = _metadata_product(metadata_file, relative_metadata)
    if error:
        return [*errors, error], None

    product_file, error = _product_file(
        metadata_file,
        product,
        filename,
        repository,
        tracked,
        relative_metadata,
    )
    if error:
        return [*errors, error], None
    if product_file is None:
        return errors, None

    product_id = product.get("id")
    if product_id is None:
        return errors, None
    digest = _file_digest(product_file)
    return errors, (str(product_id), relative_metadata, digest)


def _validate_metadata_references(
    repository,
    metadata_roots,
    require_git_tracking,
):
    """Validate metadata product references and product identifier contents."""
    tracked = _tracked_files(repository) if require_git_tracking else None
    metadata_files = sorted(
        path
        for root in metadata_roots
        if (repository / root).is_dir()
        for path in (repository / root).rglob("*.meta.yml")
        if path.is_file()
    )
    errors = []
    products_by_id = {}

    for metadata_file in metadata_files:
        file_errors, record = _validate_metadata_file(
            metadata_file,
            repository,
            tracked,
        )
        errors.extend(file_errors)
        if record is not None:
            product_id, metadata_path, digest = record
            products_by_id.setdefault(product_id, []).append((metadata_path, digest))

    for product_id, records in products_by_id.items():
        if len({digest for _, digest in records}) > 1:
            locations = ", ".join(str(metadata_file) for metadata_file, _ in records)
            errors.append(
                f"product ID {product_id} identifies different file contents: {locations}"
            )
    return errors


def _missing_scan_roots(repository, metadata_roots, workflow_root):
    """Return errors for configured scan roots that are not directories."""
    roots = [(root, "metadata") for root in metadata_roots]
    if workflow_root not in metadata_roots:
        roots.append((workflow_root, "workflow"))
    return [
        f"{root}: {kind} scan directory does not exist"
        for root, kind in roots
        if not (repository / root).is_dir()
    ]


def _submission_key(application, relative_config):
    """Return the submitted parameter version key for one application."""
    if not isinstance(application, dict) or application.get("application") != _SUBMIT_APPLICATION:
        return None, None
    try:
        configuration = application["configuration"]
        key = tuple(
            str(configuration[name]) for name in ("instrument", "parameter", "parameter_version")
        )
    except KeyError, TypeError:
        return None, (f"{relative_config}: submit application is missing configuration fields")
    return key, None


def _skip_workflow_errors(skip_file, repository):
    """Return whether a workflow is skipped and marker validation errors."""
    if skip_file.is_file():
        reason = skip_file.read_text(encoding="utf-8").strip()
        if not reason or "\n" in reason:
            return True, [f"{skip_file.relative_to(repository)}: provide a single-line reason"]
        return True, []
    return False, []


def _workflow_applications(config_file, relative_config):
    """Read and validate the applications list from one workflow file."""
    try:
        applications = ascii_handler.collect_data_from_file(config_file)["applications"]
    except (OSError, TypeError, KeyError, ValueError, yaml.YAMLError) as exc:
        return None, [f"{relative_config}: cannot read applications: {exc}"]
    if not isinstance(applications, list):
        return None, [f"{relative_config}: applications must be a list"]
    return applications, []


def _workflow_submission_keys(config_file, repository):
    """Return submitted parameter version keys from one active workflow."""
    relative_config = config_file.relative_to(repository)
    skipped, errors = _skip_workflow_errors(config_file.parent / _SKIP_MARKER, repository)
    if skipped:
        return [], errors

    applications, errors = _workflow_applications(config_file, relative_config)
    if errors:
        return [], errors

    keys = []
    for application in applications:
        key, error = _submission_key(application, relative_config)
        if error is not None:
            errors.append(error)
        elif key is not None:
            keys.append(key)
    return keys, errors


def _active_workflow_error(active_versions, key, relative_config):
    """Record one active workflow version or return its duplicate error."""
    previous = active_versions.get(key)
    if previous is None:
        active_versions[key] = relative_config
        return None
    return (
        f"{relative_config} and {previous}: active workflows share "
        f"(instrument, parameter, parameter_version) {key}"
    )


def _validate_workflow_policy(
    repository,
    workflow_root,
):
    """Validate workflow skip markers and version uniqueness."""
    workflow_directory = repository / workflow_root
    config_files = []
    if workflow_directory.is_dir():
        config_files = sorted(workflow_directory.rglob("config.yml"))
    errors = []
    active_versions = {}

    for config_file in config_files:
        keys, workflow_errors = _workflow_submission_keys(
            config_file,
            repository,
        )
        errors.extend(workflow_errors)
        relative_config = config_file.relative_to(repository)
        for key in keys:
            if error := _active_workflow_error(
                active_versions,
                key,
                relative_config,
            ):
                errors.append(error)
    return errors


def validate_repository_consistency(
    repository, metadata_roots, workflow_root, require_git_tracking=False
):
    """Return validation errors for one data repository."""
    repository = Path(repository).resolve()
    metadata_roots = tuple(Path(root) for root in metadata_roots)
    workflow_root = Path(workflow_root)
    return [
        *_missing_scan_roots(repository, metadata_roots, workflow_root),
        *_validate_metadata_references(
            repository,
            metadata_roots,
            require_git_tracking,
        ),
        *_validate_workflow_policy(repository, workflow_root),
    ]


def main():
    """Validate repository-wide metadata and workflow invariants."""
    app_context = APPLICATION.start()
    args = app_context.args
    errors = validate_repository_consistency(
        args["repository"],
        args["metadata_roots"],
        args["workflow_root"],
        require_git_tracking=args["require_git_tracking"],
    )
    if errors:
        for error in errors:
            app_context.logger.error(error)
        raise SystemExit(1)
    app_context.logger.info("Repository metadata references and workflow invariants are valid")


if __name__ == "__main__":
    main()
