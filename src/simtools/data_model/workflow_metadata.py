"""Utilities for workflow-level metadata propagation into model-parameter metadata files."""

import logging
from copy import deepcopy
from pathlib import Path

import simtools.utils.general as gen
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io import ascii_handler

logger = logging.getLogger(__name__)


def build_workflow_activity_metadata(
    args_dict,
    workflow_activity_id,
    workflow_start,
    workflow_end,
    runtime_environment,
    workflow_context,
):
    """Build workflow activity metadata from workflow execution context.

    Parameters
    ----------
    args_dict : dict
        Workflow application arguments.
    workflow_activity_id : str
        Workflow-level activity identifier.
    workflow_start : datetime
        Start time of the workflow.
    workflow_end : datetime
        End time of the workflow.
    runtime_environment : dict or None
        Runtime environment definition used for the workflow.
    workflow_context : dict
        Context with keys 'site' and 'instrument' for the workflow.

    Returns
    -------
    dict
        Activity block to be injected into model-parameter metadata files.
    """
    metadata_args = dict(args_dict)
    metadata_args["label"] = "setting_workflow"
    metadata_args["activity_id"] = workflow_activity_id
    metadata_args["activity_start"] = workflow_start.isoformat(timespec="seconds")
    metadata_args["activity_end"] = workflow_end.isoformat(timespec="seconds")
    metadata_args["runtime_environment"] = deepcopy(runtime_environment)
    metadata_args["site"] = workflow_context.get("site")
    metadata_args["instrument"] = workflow_context.get("instrument")

    collector = MetadataCollector(metadata_args, clean_meta=False)
    activity = collector.get_top_level_metadata().get("cta", {}).get("activity", {})
    if runtime_environment is not None:
        activity["runtime_environment"] = deepcopy(runtime_environment)
    return activity


def update_model_parameter_metadata_file(
    metadata_file,
    workflow_activity,
    associated_activities,
):
    """Inject workflow metadata into a model-parameter metadata file.

    Parameters
    ----------
    metadata_file : str or Path
        Path to the model-parameter metadata file to update.
    workflow_activity : dict
        Workflow activity metadata block to set as top-level activity metadata.
    associated_activities : list
        Ordered activities associated with workflow execution.

    Returns
    -------
    None
        Function updates file in place when it exists.
    """
    metadata_path = Path(metadata_file)
    if not metadata_path.exists():
        logger.debug(f"Model-parameter metadata file does not exist: {metadata_path}")
        return

    metadata = ascii_handler.collect_data_from_file(metadata_path)
    metadata = gen.change_dict_keys_case(metadata, True)
    cta_meta = metadata.get("cta", {})
    cta_meta["activity"] = deepcopy(workflow_activity)

    context = cta_meta.setdefault("context", {})
    context_associated = context.get("associated_activities") or []
    context["associated_activities"] = _merge_associated_activities(
        context_associated,
        associated_activities or [],
    )

    metadata["cta"] = cta_meta
    ascii_handler.write_data_to_file(metadata, metadata_path)
    logger.info(f"Updated workflow metadata in {metadata_path}")


def _merge_associated_activities(existing_activities, new_activities):
    """Merge associated activities preserving order and uniqueness."""
    merged_activities = []
    seen = set()
    for activity in [*(existing_activities or []), *(new_activities or [])]:
        key = (activity.get("activity_name"), activity.get("activity_id"))
        if key in seen:
            continue
        seen.add(key)
        merged_activities.append(activity)
    return merged_activities
