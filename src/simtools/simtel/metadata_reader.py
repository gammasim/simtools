"""Read sim_telarray metadata as format-neutral input-metadata fields."""

import logging

from simtools.constants import SIM_TELARRAY_META_PARAMETER_REGISTRY
from simtools.simtel import sim_telarray_metadata
from simtools.simtel.simtel_io_metadata import read_sim_telarray_metadata
from simtools.utils import names

_logger = logging.getLogger(__name__)


def read_metadata(metadata_file_name):
    """Return input-metadata fields of a sim_telarray file (see ``MetadataCollector``)."""
    global_metadata, telescope_metadata = read_sim_telarray_metadata(metadata_file_name)
    global_metadata = global_metadata or {}
    telescope_metadata = telescope_metadata or {}
    _logger.debug(
        "Validating sim_telarray metadata against %s", SIM_TELARRAY_META_PARAMETER_REGISTRY
    )
    sim_telarray_metadata.validate_metadata_values(global_metadata)
    for metadata in telescope_metadata.values():
        sim_telarray_metadata.validate_metadata_values(metadata)
    site = _get_valid_site(global_metadata.get("site_config_name"))
    return {
        "software_name": "sim_telarray",
        "software_version": global_metadata.get("simtools_simtel_tag"),
        "site": site,
        "instrument_id": global_metadata.get("array_config_name"),
        "format_name": "simtel",
        "raw_metadata": {"global": global_metadata, "telescopes": telescope_metadata},
        "associated_elements": _associated_elements(telescope_metadata, site),
    }


def _get_valid_site(site):
    """Return a validated CTA site or ``None`` for unavailable metadata."""
    if site is None:
        return None
    try:
        return names.validate_site_name(site)
    except TypeError, ValueError:
        return None


def _associated_elements(telescope_metadata, site):
    """Convert sim_telarray telescope metadata into CTA instrument entries."""
    associated_elements = []
    for metadata in telescope_metadata.values():
        telescope_name = metadata.get("optics_config_variant") or metadata.get(
            "camera_config_variant"
        )
        if telescope_name is None:
            continue
        try:
            telescope_type = names.get_array_element_type_from_name(telescope_name)
            telescope_site = names.get_site_from_array_element_name(telescope_name)
        except TypeError, ValueError:
            telescope_type = None
            telescope_site = site
        associated_elements.append(
            {
                "site": telescope_site,
                "class": "telescope",
                "type": telescope_type,
                "id": telescope_name,
            }
        )
    return associated_elements
