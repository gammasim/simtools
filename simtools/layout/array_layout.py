import logging
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import QTable

from simtools import db_handler
from simtools.data_model import data_reader
from simtools.io_operations import io_handler
from simtools.layout.geo_coordinates import GeoCoordinates
from simtools.layout.telescope_position import TelescopePosition
from simtools.utils import names
from simtools.utils.general import collect_data_from_file_or_dict

__all__ = ["InvalidTelescopeListFile", "ArrayLayout"]


class InvalidTelescopeListFile(Exception):
    """Exception for invalid telescope list file."""


class InvalidCoordinateDataType(Exception):
    """Exception for low-precision coordinate data type."""


class ArrayLayout:
    """
    Manage telescope positions at the array layout level.

    Parameters
    ----------
    mongo_db_config: dict
        MongoDB configuration.
    site: str
        Site name or location (e.g., North/South or LaPalma/Paranal)
    label: str
        Instance label.
    name: str
        Name of the layout.
    layout_center_data: dict
        Dict describing array center coordinates.
    corsika_telescope_data: dict
        Dict describing CORSIKA telescope parameters.
    telescope_list_file: str or Path
        Path to the telescope list file.
    """

    def __init__(
        self,
        mongo_db_config=None,
        site=None,
        label=None,
        name=None,
        layout_center_data=None,
        corsika_telescope_data=None,
        telescope_list_file=None,
        telescope_list_metadata_file=None,
        validate=False,
    ):
        """
        Initialize ArrayLayout.
        """

        self._logger = logging.getLogger(__name__)

        self.mongo_db_config = mongo_db_config
        self.label = label
        self.name = name
        self.site = None if site is None else names.validate_site_name(site)
        self.io_handler = io_handler.IOHandler()
        self.geo_coordinates = GeoCoordinates()

        self.telescope_list_file = None
        self._telescope_list = []
        self._epsg = None

        if telescope_list_file is None:
            self._initialize_coordinate_systems(layout_center_data)
            self._initialize_corsika_telescope(corsika_telescope_data)
        else:
            self.initialize_array_layout_from_telescope_file(
                telescope_list_file=telescope_list_file,
                telescope_list_metadata_file=telescope_list_metadata_file,
                validate=validate,
            )

    @classmethod
    def from_array_layout_name(cls, mongo_db_config, array_layout_name, label=None):
        """
        Read telescope list from file for given layout name (e.g. South-4LST, North-Prod5, ...).
        Layout definitions are given in the `data/layout` path.

        Parameters
        ----------
        mongo_db_config: dict
            MongoDB configuration.
        array_layout_name: str
            e.g. South-4LST, North-Prod5 ...
        label: str
            Instance label. Important for output file naming.

        Returns
        -------
        ArrayLayout
            Instance of the ArrayLayout.
        """

        split_name = array_layout_name.split("-")
        site_name = names.validate_site_name(split_name[0])
        array_name = names.validate_array_layout_name(split_name[1])
        valid_array_layout_name = site_name + "-" + array_name

        layout = cls(
            site=site_name,
            mongo_db_config=mongo_db_config,
            name=valid_array_layout_name,
            label=label,
        )

        telescope_list_file = layout.io_handler.get_input_data_file(
            "layout", f"telescope_positions-{valid_array_layout_name}.ecsv"
        )
        layout.initialize_array_layout_from_telescope_file(telescope_list_file)

        return layout

    def __len__(self):
        """
        Return number of telescopes in the layout.
        """
        return len(self._telescope_list)

    def __getitem__(self, i):
        """
        Return telescope at list position i.

        """
        return self._telescope_list[i]

    def _initialize_corsika_telescope(self, corsika_dict=None):
        """
        Initialize Dictionary for CORSIKA telescope parameters. Allow input from different sources
        (dictionary, yaml, ecsv header), which require checks to handle units correctly.

        Parameters
        ----------
        corsika_dict dict
            dictionary with CORSIKA telescope parameters

        """
        self._corsika_telescope = {}

        if corsika_dict is None:
            self._logger.debug("Initialize CORSIKA telescope parameters from file")
            corsika_dict = self._from_corsika_file_to_dict()
        else:
            self._logger.debug(f"Initialize CORSIKA telescope parameters from dict: {corsika_dict}")

        self._initialize_corsika_telescope_from_dict(corsika_dict)

    def _from_corsika_file_to_dict(self, file_name=None):
        """
        Get the corsika parameter file and return a dictionary with the keys necessary to
        initialize this class.

        Parameters
        ----------
        file_name: str or Path
            File from which to extract the corsika parameters. Default is
            data/parameters/corsika_parameters.yml

        Returns
        ------
        corsika_dict:
            Dictionary with corsika telescopes information.

        Raises
        ------
        FileNotFoundError:
            If file_name does not exist.
        """
        if file_name is None:
            try:
                corsika_parameters_dict = collect_data_from_file_or_dict(
                    self.io_handler.get_input_data_file("parameters", "corsika_parameters.yml"),
                    None,
                )
            except io_handler.IncompleteIOHandlerInit:
                self._logger.info("Error reading CORSIKA parameters from file")
                return {}
        else:
            if not isinstance(file_name, Path):
                file_name = Path(file_name)
            if file_name.exists():
                corsika_parameters_dict = collect_data_from_file_or_dict(file_name, None)
            else:
                raise FileNotFoundError

        corsika_dict = {}
        corsika_pars = ["corsika_sphere_radius", "corsika_sphere_center"]
        for simtools_par in corsika_pars:
            corsika_par = names.translate_simtools_to_corsika(simtools_par)
            corsika_dict[simtools_par] = {}
            for key, value in corsika_parameters_dict[corsika_par].items():
                corsika_dict[simtools_par][key] = value["value"]
                try:
                    unit = value["unit"]
                    corsika_dict[simtools_par][key] = corsika_dict[simtools_par][key] * u.Unit(unit)
                except KeyError:
                    self._logger.warning(
                        "Key not valid. Dictionary does not have a key 'unit'. Continuing without "
                        "the unit."
                    )

        if self.mongo_db_config is None:
            self._logger.error("DB connection info was not provided, cannot set site altitude")
            raise ValueError
        if self.site is None:
            self._logger.error("Site was not provided, cannot set site altitude")
            raise ValueError

        db = db_handler.DatabaseHandler(mongo_db_config=self.mongo_db_config)
        self._logger.debug("Reading site parameters from DB")
        _site_pars = db.get_site_parameters(self.site, "Released", only_applicable=True)
        corsika_dict["corsika_obs_level"] = _site_pars["altitude"]["Value"] * u.Unit(
            _site_pars["altitude"]["units"]
        )

        return corsika_dict

    def _initialize_sphere_parameters(self, sphere_dict):
        """
        Set CORSIKA sphere parameters from dictionary. Type of input varies and depend on data \
        source for these parameters.

        Example for sphere_dict: {LST: 12.5 m, MST: 9.15 m, SST: 3 m}

        Parameters
        ----------
        sphere_dict: dict
            dictionary with sphere parameters

        Returns
        -------
        dict
            dictionary with sphere parameters.

        """

        _sphere_dict_cleaned = {}
        try:
            for key, value in sphere_dict.items():
                if isinstance(value, (str, u.Quantity)):
                    _sphere_dict_cleaned[key] = u.Quantity(value)
                else:
                    _sphere_dict_cleaned[key] = value["value"] * u.Unit(value["unit"])
        except (TypeError, KeyError) as exc:
            self._logger.error(f"Error setting CORSIKA sphere parameters from {sphere_dict}")
            raise exc

        return _sphere_dict_cleaned

    def _initialize_corsika_telescope_from_dict(self, corsika_dict):
        """
        Initialize CORSIKA telescope parameters from a dictionary.

        Parameters
        ----------
        corsika_dict dict
            dictionary with CORSIKA telescope parameters

        """

        try:
            self._corsika_telescope["corsika_obs_level"] = u.Quantity(
                corsika_dict["corsika_obs_level"]
            )
        except (TypeError, KeyError):
            self._corsika_telescope["corsika_obs_level"] = np.nan * u.m

        for key in ["corsika_sphere_center", "corsika_sphere_radius"]:
            try:
                self._corsika_telescope[key] = self._initialize_sphere_parameters(corsika_dict[key])
            except (TypeError, KeyError):
                pass

    def _initialize_coordinate_systems(self, center_dict=None):
        """
        Initialize array center and coordinate systems.
        By definition, the array center is at (0,0) in
        the ground coordinate system.

        Parameters
        ----------
        center_dict: dict
            dictionary with coordinates of array center.

        Raises
        ------
        TypeError
            invalid array center definition.

        """

        center_dict = {} if center_dict is None else center_dict

        self._array_center = TelescopePosition()
        self._array_center.name = "array_center"
        self._array_center.set_coordinates("ground", 0.0 * u.m, 0.0 * u.m, 0.0 * u.m)
        self._set_array_center_mercator(center_dict)
        self._set_array_center_utm(center_dict)
        self._array_center.set_altitude(u.Quantity(center_dict.get("center_alt", np.nan * u.m)))
        _name = center_dict.get("array_name")
        self.name = _name if _name is not None else self.name

        self._array_center.convert_all(
            crs_local=self.geo_coordinates.crs_local(self._array_center),
            crs_wgs84=self.geo_coordinates.crs_wgs84(),
            crs_utm=self.geo_coordinates.crs_utm(self._epsg),
        )

    def _set_array_center_mercator(self, center_dict):
        """
        Set array center coordinates in mercator system.

        """

        try:
            self._array_center.set_coordinates(
                "mercator",
                u.Quantity(center_dict.get("center_lat", np.nan * u.deg)),
                u.Quantity(center_dict.get("center_lon", np.nan * u.deg)),
            )
        except TypeError:
            pass

    def _set_array_center_utm(self, center_dict):
        """
        Set array center coordinates in UTM system.
        Convert array center position to WGS84 system
        (as latitudes are required for the definition
        for the definition of the ground coordinate system)

        """
        try:
            self._epsg = center_dict.get("EPSG", None)
            self._array_center.set_coordinates(
                "utm",
                u.Quantity(center_dict.get("center_easting", np.nan * u.m)),
                u.Quantity(center_dict.get("center_northing", np.nan * u.m)),
            )
            self._array_center.convert_all(
                crs_local=None,
                crs_wgs84=self.geo_coordinates.crs_wgs84(),
                crs_utm=self.geo_coordinates.crs_utm(self._epsg),
            )
        except TypeError:
            pass

    def _altitude_from_corsika_z(self, pos_z=None, altitude=None, tel_name=None):
        """
        Calculate altitude from CORSIKA z-coordinate (if pos_z is given) or CORSIKA z-coordinate \
        from altitude (if altitude is given).

        Parameters
        ----------
        pos_z: astropy.Quantity
            CORSIKA z-coordinate of telescope in equivalent units of meter.
        altitude: astropy.Quantity
            Telescope altitude in equivalent units of meter.
        tel_name: str
            Telescope Name.

        Returns
        -------
        astropy.Quantity
            Altitude or CORSIKA z-coordinate (np.nan in case of ill-defined value).

        """
        if pos_z is not None and altitude is None:
            return TelescopePosition.convert_telescope_altitude_from_corsika_system(
                pos_z,
                self._corsika_telescope["corsika_obs_level"],
                self._get_corsika_sphere_center(tel_name),
            )

        if altitude is not None and pos_z is None:
            return TelescopePosition.convert_telescope_altitude_to_corsika_system(
                altitude,
                self._corsika_telescope["corsika_obs_level"],
                self._get_corsika_sphere_center(tel_name),
            )
        return np.nan

    def _get_corsika_sphere_center(self, tel_name):
        """
        Return CORSIKA sphere center value for given telescope.

        Parameters
        ----------
        tel_name: str
            Telescope Name.

        Returns
        -------
        astropy.Quantity
            Telescope sphere center value (0.0*u.m if sphere center is not defined).

        Raises
        ------
        KeyError
            if Missing definition of CORSIKA sphere center for this telescope type.

        """

        try:
            return self._corsika_telescope["corsika_sphere_center"][
                names.get_telescope_class(tel_name)
            ]
        except KeyError:
            self._logger.warning(
                "Missing definition of CORSIKA sphere center for telescope "
                f"{tel_name} of type {names.get_telescope_class(tel_name)}"
            )
        except ValueError:
            self._logger.warning(
                f"Missing definition of CORSIKA sphere center for telescope {tel_name}"
            )

        return 0.0 * u.m

    def _load_telescope_names(self, row):
        """
        Read and set telescope names.

        Parameters
        ----------
        row: astropy table row
            table row.

        Returns
        -------
        tel: TelescopePosition
            Instance of TelescopePosition.

        Raises
        ------
        InvalidTelescopeListFile
            in case neither telescope name or asset_code / sequence number are given.

        """

        tel = TelescopePosition()
        try:
            tel.name = row["telescope_name"]
            if "asset_code" not in row:
                try:
                    tel.asset_code = names.get_telescope_class(tel.name)
                # asset code is not a valid telescope name; possibly a calibration device
                except ValueError:
                    tel.asset_code = tel.name.split("-")[0]
        except KeyError:
            pass
        try:
            if tel.name is None:
                tel.name = row["asset_code"] + "-" + row["sequence_number"]
            tel.asset_code = row["asset_code"]
            tel.sequence_number = row["sequence_number"]
        except KeyError:
            pass
        try:
            tel.geo_code = row["geo_code"]
        except KeyError:
            pass
        if tel.name is None:
            msg = "Missing required row with telescope_name or asset_code/sequence_number"
            self._logger.error(msg)
            raise InvalidTelescopeListFile(msg)

        return tel

    def _assign_unit_to_quantity(self, value, unit):
        """
        Assign unit to quantity.

        Parameters
        ----------
        value:
            value to get a unit. It can be a float, int, or a Quantity (convertible to 'unit').
        unit: astropy.units.Unit
            Unit to apply to 'quantity'.

        Returns
        -------
        astropy.units.Quantity
            Quantity of value 'quantity' and unit 'unit'.
        """
        if isinstance(value, u.Quantity):
            if isinstance(value.unit, type(unit)):
                return value
            try:
                value = value.to(unit)
                return value
            except u.UnitConversionError:
                self._logger.error(f"Cannot convert {value.unit} to {unit}.")
                raise
        return value * unit

    def _try_set_coordinate(self, row, tel, table, crs_name, key1, key2):
        """Function auxiliary to self._load_telescope_list. It sets the coordinates.

        Parameters
        ----------
        row: dict
            A row of the astropy.table.Table with array element coordinates.
        tel: TelescopePosition
            Instance of TelescopePosition.
        table: astropy.table.Table or astropy.table.QTable
            data table with array element coordinates.
        crs_name: str
            Name of coordinate system.
        key1: str
            Name of x-coordinate.
        key2: str
            Name of y-coordinate.
        """
        try:
            tel.set_coordinates(
                crs_name,
                self._assign_unit_to_quantity(row[key1], table[key1].unit),
                self._assign_unit_to_quantity(row[key2], table[key2].unit),
            )
        except KeyError:
            pass

    def _try_set_altitude(self, row, tel, table):
        """
        Function auxiliary to self._load_telescope_list. It sets the altitude of the
        TelescopePosition instance.

        Parameters
        ----------
        row: dict
            A row of the astropy.table.Table with array element coordinates.
        tel: TelescopePosition
            Instance of TelescopePosition.
        table: astropy.table.Table or astropy.table.QTable
            data table with array element coordinates.
        """
        try:
            tel.set_altitude(
                self._altitude_from_corsika_z(
                    pos_z=self._assign_unit_to_quantity(
                        row["position_z"], table["position_z"].unit
                    ),
                    tel_name=tel.name,
                )
            )
        except KeyError:
            pass
        try:
            tel.set_altitude(self._assign_unit_to_quantity(row["altitude"], table["altitude"].unit))
        except KeyError:
            pass

    def _load_telescope_list(self, table):
        """
        Load list of telescope from an astropy table (support both QTable and Table)

        Parameters
        ----------
        table: astropy.table.Table or astropy.table.QTable
            data table with array element coordinates

        """
        for row in table:
            tel = self._load_telescope_names(row)
            self._try_set_coordinate(row, tel, table, "ground", "position_x", "position_y")
            self._try_set_coordinate(row, tel, table, "utm", "utm_east", "utm_north")
            self._try_set_coordinate(row, tel, table, "mercator", "latitude", "longitude")
            self._try_set_altitude(row, tel, table)

            self._telescope_list.append(tel)

    def initialize_array_layout_from_telescope_file(
        self, telescope_list_file, telescope_list_metadata_file=None, validate=False
    ):
        """
        Initialize the Layout array from a telescope list file.

        Parameters
        ----------
        telescope_list_file: str or Path
            Path to the telescope list file.
        telescope_list_metadata_file: str or Path
            Path to the telescope list metadata file.
        validate: bool
            Validate the telescope list file.

        Returns
        -------
        astropy.table.QTable
            Table with the telescope layout information.
        """
        table = data_reader.read_table_from_file(
            file_name=telescope_list_file,
            validate=validate,
            metadata_file=telescope_list_metadata_file,
        )
        self._initialize_corsika_telescope(table.meta)
        self._initialize_coordinate_systems(table.meta)
        self._load_telescope_list(table)

        return table

    def add_telescope(self, telescope_name, crs_name, xx, yy, altitude=None, tel_corsika_z=None):
        """
        Add an individual telescope to the telescope list.

        Parameters
        ----------
        telescope_name: str
            Name of the telescope starting with L, M or S (e.g. LST-01, MST-06 ...)
        crs_name: str
            Name of coordinate system
        xx: astropy.units.quantity.Quantity
            x-coordinate for the given coordinate system
        yy: astropy.units.quantity.Quantity
            y-coordinate for the given coordinate system
        altitude: astropy.units.quantity.Quantity
            Altitude coordinate in equivalent units of u.m.
        tel_corsika_z: astropy.units.quantity.Quantity
            CORSIKA z-position (requires setting of CORSIKA observation level and telescope sphere\
            center).
        """

        tel = TelescopePosition(name=telescope_name)
        tel.set_coordinates(crs_name, xx, yy)
        if altitude is not None:
            tel.set_altitude(altitude)
        elif tel_corsika_z is not None:
            tel.set_altitude(self._altitude_from_corsika_z(pos_z=tel_corsika_z, tel_name=tel.name))
        self._telescope_list.append(tel)

    def _get_export_metadata(self, export_corsika_meta=False):
        """
        File metadata for export of array element list to file. Included array center definition,\
        CORSIKA telescope parameters, and EPSG center

        Parameters
        ----------
        export_corsika_meta: bool
            write CORSIKA metadata.

        Returns
        -------
        dict
            Metadata header for array element list export.

        """

        _meta = {
            "center_lon": None,
            "center_lat": None,
            "center_northing": None,
            "center_easting": None,
            "center_alt": None,
        }
        if self._array_center:
            _meta["center_lat"], _meta["center_lon"], _ = self._array_center.get_coordinates(
                "mercator"
            )
            (
                _meta["center_easting"],
                _meta["center_northing"],
                _meta["center_alt"],
            ) = self._array_center.get_coordinates("utm")
        if export_corsika_meta:
            _meta.update(self._corsika_telescope)
        _meta["EPSG"] = self._epsg
        _meta["array_name"] = self.name

        return _meta

    def export_telescope_list_table(self, crs_name, corsika_z=False):
        """
        Export array elements positions to astropy table.

        Parameters
        ----------
        crs_name: str
            Name of coordinate system to be used for export.
        corsika_z: bool
            Write telescope height in CORSIKA coordinates (for CORSIKA system).

        Returns
        -------
        astropy.table.QTable
            Astropy table with the telescope layout information.

        """

        table = QTable(meta=self._get_export_metadata(crs_name == "ground"))

        tel_names, asset_code, sequence_number, geo_code = [], [], [], []
        pos_x, pos_y, pos_z = [], [], []
        for tel in self._telescope_list:
            tel_names.append(tel.name)
            asset_code.append(tel.asset_code)
            sequence_number.append(tel.sequence_number)
            geo_code.append(tel.geo_code)
            x, y, z = tel.get_coordinates(crs_name)
            if corsika_z:
                z = self._altitude_from_corsika_z(altitude=z, tel_name=tel.name)
            pos_x.append(x)
            pos_y.append(y)
            pos_z.append(z)

        # prefer asset_code / sequence_number of telescope_name
        if all(v is not None for v in asset_code) and all(v is not None for v in sequence_number):
            table["asset_code"] = asset_code
            table["sequence_number"] = sequence_number
        else:
            table["telescope_name"] = tel_names
        if any(v is not None for v in geo_code):
            table["geo_code"] = geo_code

        try:
            _name_x, _name_y, _name_z = self._telescope_list[0].get_coordinates(
                crs_name=crs_name, coordinate_field="name"
            )
            table[_name_x] = pos_x
            table[_name_y] = pos_y
            if corsika_z:
                table["position_z"] = pos_z
            else:
                table[_name_z] = pos_z
        except IndexError:
            pass

        return table

    def get_number_of_telescopes(self):
        """
        Return the number of telescopes in the list.

        Returns
        -------
        int
            Number of telescopes.
        """
        return len(self._telescope_list)

    def get_corsika_input_list(self):
        """
        Get a string with the piece of text to be added to the CORSIKA input file.

        Returns
        -------
        str
            Piece of text to be added to the CORSIKA input file.

        Raises
        ------
        KeyError
            if Missing definition of CORSIKA sphere radius or obs_level.
        """

        corsika_list = ""
        for tel in self._telescope_list:
            pos_x, pos_y, pos_z = tel.get_coordinates("ground")
            try:
                sphere_radius = self._corsika_telescope["corsika_sphere_radius"][
                    names.get_telescope_class(tel.name)
                ]
            except KeyError:
                self._logger.error("Missing definition of CORSIKA sphere radius")
                raise
            try:
                pos_z = tel.convert_telescope_altitude_to_corsika_system(
                    pos_z,
                    self._corsika_telescope["corsika_obs_level"],
                    self._get_corsika_sphere_center(tel.name),
                )
            except KeyError:
                self._logger.error("Missing definition of CORSIKA sphere center / obs_level")
                raise

            corsika_list += "TELESCOPE"
            for pos in [pos_x, pos_y, pos_z]:
                corsika_list += f"\t {pos.value:.3f}E2"
            corsika_list += f"\t {sphere_radius.value:.3f}E2"
            corsika_list += f"\t # {tel.name}\n"

        return corsika_list

    def _print_all(self):
        """ "
        Print all columns for all coordinate systems.

        """

        print(f"ArrayLayout: {self.name}")
        print("ArrayCenter")
        print(self._array_center)
        print("Telescopes")
        for tel in self._telescope_list:
            print(tel)

    def _print_compact(self, compact_printing, corsika_z=False):
        """
        Compact printing of list of telescopes.


        Parameters
        ----------
        compact_printing: str
            Compact printout for a single coordinate system. Coordinates in all systems are \
            printed, if compact_printing is None.
        corsika_z: bool
            Print telescope height in CORSIKA coordinates (for CORSIKA system)

        """

        for tel in self._telescope_list:
            if corsika_z:
                _corsika_obs_level = self._corsika_telescope["corsika_obs_level"]
                _corsika_sphere_center = self._get_corsika_sphere_center(tel.name)
            else:
                _corsika_obs_level = None
                _corsika_sphere_center = None

            tel.print_compact_format(
                crs_name=compact_printing,
                print_header=(tel == self._telescope_list[0]),
                corsika_obs_level=_corsika_obs_level,
                corsika_sphere_center=_corsika_sphere_center,
            )

    def print_telescope_list(self, compact_printing="", corsika_z=False):
        """
        Print list of telescopes in latest released layout.

        Parameters
        ----------
        compact_printing: str
            Compact printout for a single coordinate system.
        corsika_z: bool
            Print telescope height in CORSIKA coordinates (for CORSIKA system).
        """

        if len(compact_printing) == 0:
            self._print_all()
        else:
            self._print_compact(compact_printing, corsika_z)

    def convert_coordinates(self):
        """Perform all the possible conversions the coordinates of the tel positions."""

        self._logger.info("Converting telescope coordinates")

        crs_wgs84 = self.geo_coordinates.crs_wgs84()
        crs_local = self.geo_coordinates.crs_local(self._array_center)
        crs_utm = self.geo_coordinates.crs_utm(self._epsg)

        for tel in self._telescope_list:
            tel.convert_all(
                crs_local=crs_local,
                crs_wgs84=crs_wgs84,
                crs_utm=crs_utm,
            )

    @staticmethod
    def include_radius_into_telescope_table(telescope_table):
        """
        Include the radius of the telescopes types into the astropy.table.QTable telescopes_table

        Parameters
        ----------
        telescope_table: astropy.QTable
            Astropy QTable with telescope information.

        Returns
        -------
        astropy.QTable
            Astropy QTable with telescope information updated with the radius.
        """

        telescope_table["radius"] = [
            u.Quantity(
                telescope_table.meta["corsika_sphere_radius"][
                    names.get_telescope_class(tel_name_now)
                ]
            )
            .to("m")
            .value
            for tel_name_now in telescope_table["telescope_name"]
        ]
        telescope_table["radius"] = telescope_table["radius"].quantity * u.m
        return telescope_table

    def select_assets(self, asset_list=None):
        """
        Select a subsets of telescopes / assets from the layout.

        Parameters
        ----------
        asset_list: list
            List of assets to be selected.

        Raises
        ------
        ValueError
            If the asset list is empty.

        """

        _n_telescopes = len(self._telescope_list)
        try:
            if len(asset_list) > 0:
                self._telescope_list = [
                    tel for tel in self._telescope_list if tel.asset_code in asset_list
                ]
            self._logger.info(
                f"Selected {len(self._telescope_list)} telescopes"
                f" (from originally {_n_telescopes})"
            )
        except TypeError:
            self._logger.info("No asset list provided, keeping all telescopes")
