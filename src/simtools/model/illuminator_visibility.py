"""Module for reading and parsing illuminator-telescope visibility tables."""

import logging

_logger = logging.getLogger(__name__)


class IlluminatorTelescopeVisibility:
    """
    Reader and parser for illuminator-telescope visibility tables.

    Parses visibility data defining which telescopes can be illuminated by each
    illuminator, accounting for shadowing, blocking, topography, and distance
    constraints.

    The input is a list of structured records. Schema version 0.1.0 model
    records store the same data as a ``columns``/``rows`` mapping and are
    normalized before parsing. The record structure is:

    .. code-block:: python

        [
            {"illuminator_id": "ILLN-01", "telescope_id": "LSTN-01", "visible": false},
            {"illuminator_id": "ILLN-01", "telescope_id": "MSTN-02", "visible": true},
        ]

    Parameters
    ----------
    visibility_data : list of dict
        One record per illuminator-telescope combination.

    Raises
    ------
    ValueError
        If the data format is invalid or missing required columns.
    """

    def __init__(self, visibility_data):
        """Initialize the visibility table reader."""
        _logger.info("Reading illuminator visibility table")
        self._parse_visibility_data(visibility_data)
        self._validate_data()
        self._valid_pairs = [(ill, tel) for (ill, tel), vis in self._pairs.items() if vis]
        _logger.info(
            f"Found {len(self._valid_pairs)} valid illuminator-telescope pairs "
            f"({len(self._illuminators)} illuminators x {len(self._telescopes)} telescopes)"
        )

    def _parse_visibility_data(self, visibility_data):
        """
        Parse visibility records into internal lookup structures.

        Parameters
        ----------
        visibility_data : list of dict
            One structured record per illuminator-telescope pair.
        """
        visibility_data = _normalize_visibility_records(visibility_data)
        if not isinstance(visibility_data, list):
            raise ValueError(f"Expected a list of visibility records, got {type(visibility_data)}")

        # Build internal data structures
        self._illuminators = []
        self._telescopes = []
        self._pairs = {}  # (illuminator, telescope) -> bool

        for row in visibility_data:
            try:
                illuminator = row["illuminator_id"]
                telescope = row["telescope_id"]
                visible = bool(row["visible"])
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    "Visibility records must contain illuminator_id, telescope_id, and visible"
                ) from exc

            if illuminator not in self._illuminators:
                self._illuminators.append(illuminator)
            if telescope not in self._telescopes:
                self._telescopes.append(telescope)

            self._pairs[(illuminator, telescope)] = visible

        _logger.debug(
            f"Parsed visibility table with {len(self._illuminators)} illuminators "
            f"and {len(self._telescopes)} telescopes"
        )

    def _validate_data(self):
        """
        Validate that the parsed data is consistent.

        Raises
        ------
        ValueError
            If the data has no illuminators or no telescopes.
        """
        if len(self._illuminators) == 0:
            raise ValueError("Visibility table contains no illuminators")
        if len(self._telescopes) == 0:
            raise ValueError("Visibility table contains no telescopes")

    def get_telescopes(self):
        """
        Get list of all telescopes in the visibility table.

        Returns
        -------
        list of str
            List of telescope identifiers.
        """
        return list(self._telescopes)

    def get_valid_pairs(self):
        """
        Get all valid illuminator-telescope pairs.

        A pair is valid if the visibility value is True (telescope is illuminated
        and not shadowed/blocked).

        Returns
        -------
        list of tuple
            List of (illuminator_id, telescope_id) tuples for all valid pairs.
        """
        return list(self._valid_pairs)

    @property
    def n_telescopes(self):
        """Get the number of telescopes in the table."""
        return len(self._telescopes)

    @property
    def n_valid_pairs(self):
        """Get the total number of valid illuminator-telescope pairs."""
        return len(self._valid_pairs)


def _normalize_visibility_records(visibility_data):
    """Return structured records from a schema 0.1 row-table mapping."""
    if not isinstance(visibility_data, dict):
        return visibility_data

    columns = visibility_data.get("columns")
    rows = visibility_data.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError("Visibility row table must contain list-valued 'columns' and 'rows'")
    if any(not isinstance(row, list) or len(row) != len(columns) for row in rows):
        raise ValueError("Visibility row table rows must match the declared columns")
    return [dict(zip(columns, row, strict=True)) for row in rows]
