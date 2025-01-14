"""Retrieval of array elements from the database."""

from functools import lru_cache

from pymongo import ASCENDING


@lru_cache
def get_array_elements(db_collection, model_version):
    """
    Get all array element names and their design model for a given DB collection.

    Uses the 'design_model' parameter to determine the design model of the array element.
    Assumes that a design model is defined for every array element.

    Parameters
    ----------
    db_collection:
        pymongo.collection.Collection
    model_version: str
       Model version.

    Returns
    -------
    dict
        Dict with array element names found and their design model

    Raises
    ------
    ValueError
        If query for collection name not implemented.
    KeyError
        If array element entry in the database is incomplete.

    """
    query = {"version": model_version}
    results = db_collection.find(query, {"instrument": 1, "value": 1, "parameter": 1}).sort(
        "instrument", ASCENDING
    )

    _all_available_array_elements = {}
    for doc in results:
        try:
            if doc["parameter"] == "design_model":
                _all_available_array_elements[doc["instrument"]] = doc["value"]
        except KeyError as exc:
            raise KeyError(f"Incomplete array element entry in the database: {doc}.") from exc

    if len(_all_available_array_elements) == 0:
        raise ValueError(f"No array elements found in DB collection {db_collection}.")

    return _all_available_array_elements
