"""Module for reading and parsing illuminator-telescope visibility tables."""

import logging

_logger = logging.getLogger(__name__)


class IlluminatorTelescopeVisibility:
    """
    Reader and parser for illuminator-telescope visibility tables.

    Parses visibility data defining which telescopes can be illuminated by each
    illuminator, accounting for shadowing, blocking, topography, and distance
    constraints.

    The input is a dict with keys "columns" and "rows", as stored in the model
    parameter JSON file. The expected structure is:

    .. code-block:: python

        {
            "columns": ["illuminator_id", "telescope_id", "visible"],
            "rows": [
                ["ILLN-01", "LSTN-01", false],
                ["ILLN-01", "MSTN-02", true],
                ...
            ]
        }

    Parameters
    ----------
    visibility_data : dict
        Dictionary with "columns" and "rows" keys containing the visibility table.

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
        Parse the visibility data dictionary into internal lookup structures.

        Parameters
        ----------
        visibility_data : dict
            Dictionary with "columns" and "rows" keys.
        """
        if not isinstance(visibility_data, dict):
            raise ValueError(
                f"Expected dict with 'columns' and 'rows', got {type(visibility_data)}"
            )
        if "columns" not in visibility_data or "rows" not in visibility_data:
            raise ValueError(
                "Visibility data must contain 'columns' and 'rows' keys. "
                f"Found keys: {list(visibility_data.keys())}"
            )

        columns = visibility_data["columns"]
        rows = visibility_data["rows"]

        # Identify column indices
        try:
            ill_idx = columns.index("illuminator_id")
            tel_idx = columns.index("telescope_id")
            vis_idx = columns.index("visible")
        except ValueError as exc:
            raise ValueError(
                "Visibility data must have columns 'illuminator_id', 'telescope_id', "
                f"and 'visible'. Found: {columns}"
            ) from exc

        # Build internal data structures
        self._illuminators = []
        self._telescopes = []
        self._pairs = {}  # (illuminator, telescope) -> bool

        for row in rows:
            illuminator = row[ill_idx]
            telescope = row[tel_idx]
            visible = bool(row[vis_idx])

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

    def get_illuminators(self):
        """
        Get list of all illuminators in the visibility table.

        Returns
        -------
        list of str
            List of illuminator identifiers.
        """
        return list(self._illuminators)

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

    def is_valid_pair(self, illuminator, telescope):
        """
        Check if a specific illuminator-telescope pair is valid.

        Parameters
        ----------
        illuminator : str
            Illuminator identifier.
        telescope : str
            Telescope identifier.

        Returns
        -------
        bool
            True if the pair is valid (telescope is illuminated), False otherwise.

        Raises
        ------
        ValueError
            If the illuminator or telescope is not found in the table.
        """
        if illuminator not in self._illuminators:
            raise ValueError(f"Illuminator '{illuminator}' not found in visibility table")
        if telescope not in self._telescopes:
            raise ValueError(f"Telescope '{telescope}' not found in visibility table")

        return self._pairs.get((illuminator, telescope), False)

    def get_telescopes_for_illuminator(self, illuminator):
        """
        Get all telescopes that can be illuminated by a specific illuminator.

        Parameters
        ----------
        illuminator : str
            Illuminator identifier.

        Returns
        -------
        list of str
            List of telescope identifiers that can be illuminated.

        Raises
        ------
        ValueError
            If the illuminator is not found in the table.
        """
        if illuminator not in self._illuminators:
            raise ValueError(f"Illuminator '{illuminator}' not found in visibility table")

        return [tel for tel in self._telescopes if self._pairs.get((illuminator, tel), False)]

    def get_illuminators_for_telescope(self, telescope):
        """
        Get all illuminators that can illuminate a specific telescope.

        Parameters
        ----------
        telescope : str
            Telescope identifier.

        Returns
        -------
        list of str
            List of illuminator identifiers that can illuminate this telescope.

        Raises
        ------
        ValueError
            If the telescope is not found in the table.
        """
        if telescope not in self._telescopes:
            raise ValueError(f"Telescope '{telescope}' not found in visibility table")

        return [ill for ill in self._illuminators if self._pairs.get((ill, telescope), False)]

    @property
    def n_illuminators(self):
        """Get the number of illuminators in the table."""
        return len(self._illuminators)

    @property
    def n_telescopes(self):
        """Get the number of telescopes in the table."""
        return len(self._telescopes)

    @property
    def n_valid_pairs(self):
        """Get the total number of valid illuminator-telescope pairs."""
        return len(self._valid_pairs)
