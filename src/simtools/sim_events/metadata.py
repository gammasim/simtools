"""Metadata helpers for reduced simulation event lists."""

from simtools import version
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io.ascii_handler import to_builtin

SIMULATION_METADATA_SCHEMA_NAME = "simtools.reduced_event_list.simulation_metadata"
SIMULATION_METADATA_SCHEMA_VERSION = "1.0.0"


def build_standard_metadata(args_dict, output_file, product_data_name="reduced_event_data"):
    """Collect standard metadata for an embedded HDF5 product."""
    metadata_args = dict(args_dict or {})
    metadata_args.update(
        {
            "output_file": str(output_file),
            "output_file_format": "HDF5",
            "metadata_product_data_name": product_data_name,
        }
    )
    metadata = MetadataCollector(args_dict=metadata_args, clean_meta=False).get_top_level_metadata()
    instrument = metadata.get("cta", {}).get("instrument", {})
    if "ID" in instrument:
        instrument["id"] = instrument.pop("ID")
    return metadata


def build_simulation_metadata(
    input_files,
    array_models=None,
    simulation_software="sim_telarray",
):
    """Build the versioned simulation metadata document for one output file.

    Parameters
    ----------
    input_files : list of dict
        Rich per-input-file provenance records from
        :class:`~simtools.sim_events.writer.EventDataWriter`.
    array_models : list of dict, optional
        Plain dictionaries exported from resolved ``ArrayModel`` instances.
    simulation_software : str, optional
        Simulation software name.

    Returns
    -------
    dict
        JSON-compatible simulation metadata document.
    """
    model_exports = [to_builtin(model) for model in (array_models or [])]
    models_status = "complete" if model_exports else "unavailable"
    metadata = {
        "schema_name": SIMULATION_METADATA_SCHEMA_NAME,
        "schema_version": SIMULATION_METADATA_SCHEMA_VERSION,
        "provenance": {
            "source": "resolved_array_models" if model_exports else "input_files_only",
            "simulation_software": {"name": simulation_software},
            "simtools_version": version.__version__,
        },
        "inputs": {"files": to_builtin(input_files)},
        "models": {
            "status": models_status,
            "arrays": model_exports,
        },
    }
    if not model_exports:
        metadata["models"]["reason"] = "No resolved ArrayModel snapshot was supplied."
    validate_simulation_metadata(metadata)
    return metadata


def validate_simulation_metadata(metadata):
    """Validate required structural fields of a simulation metadata document."""
    required = {"schema_name", "schema_version", "provenance", "inputs", "models"}
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"Simulation metadata is missing required field(s): {', '.join(missing)}")
    if metadata["schema_name"] != SIMULATION_METADATA_SCHEMA_NAME:
        raise ValueError(f"Unsupported simulation metadata schema '{metadata['schema_name']}'.")
    models = metadata["models"]
    if models.get("status") not in {"complete", "unavailable"}:
        raise ValueError("Simulation metadata models.status must be 'complete' or 'unavailable'.")
    if models["status"] == "unavailable" and models.get("arrays"):
        raise ValueError("Unavailable simulation metadata must not contain model arrays.")
    return metadata
