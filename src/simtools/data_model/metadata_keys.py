"""Top-level key of the simtools observatory metadata block."""

OBSERVATORY_KEY = "ctao"
# Top-level key of products written with metadata schema versions before 3.0.0
LEGACY_OBSERVATORY_KEY = "cta"


def get_observatory_metadata(metadata):
    """
    Return the observatory metadata block of a metadata dictionary.

    Reads the ``ctao`` block and falls back to the ``cta`` block of older products.
    Upper-case keys are accepted.

    Parameters
    ----------
    metadata: dict
        Metadata dictionary.

    Returns
    -------
    dict
        Observatory metadata block (empty if not present).
    """
    for key in (OBSERVATORY_KEY, LEGACY_OBSERVATORY_KEY):
        for spelling in (key, key.upper()):
            if spelling in metadata:
                return metadata[spelling]
    return {}


def normalize_observatory_key(metadata):
    """
    Return metadata with the ``cta`` block of older products renamed to ``ctao``.

    The case of the key is preserved (``CTA`` becomes ``CTAO``).

    Parameters
    ----------
    metadata: dict
        Metadata dictionary.

    Returns
    -------
    dict
        Metadata dictionary using the ``ctao`` key.
    """
    renamed = {}
    for key, value in metadata.items():
        if key == LEGACY_OBSERVATORY_KEY and OBSERVATORY_KEY not in metadata:
            key = OBSERVATORY_KEY
        elif key == LEGACY_OBSERVATORY_KEY.upper() and OBSERVATORY_KEY.upper() not in metadata:
            key = OBSERVATORY_KEY.upper()
        renamed[key] = value
    return renamed
