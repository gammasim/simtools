"""Top-level key of the simtools observatory metadata block."""

OBSERVATORY_KEY = "ctao"
# Top-level key of products written with metadata schema versions before 3.0.0
LEGACY_OBSERVATORY_KEY = "cta"


def get_observatory_metadata(metadata):
    """Return the ``ctao`` (or legacy ``cta``, any case) metadata block; empty if absent."""
    for key in (OBSERVATORY_KEY, LEGACY_OBSERVATORY_KEY):
        for spelling in (key, key.upper()):
            if spelling in metadata:
                return metadata[spelling]
    return {}


def normalize_observatory_key(metadata):
    """Return metadata with the legacy ``cta`` block renamed to ``ctao``."""
    if LEGACY_OBSERVATORY_KEY not in metadata or OBSERVATORY_KEY in metadata:
        return metadata
    return {
        OBSERVATORY_KEY if key == LEGACY_OBSERVATORY_KEY else key: value
        for key, value in metadata.items()
    }
