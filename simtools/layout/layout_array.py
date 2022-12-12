import logging
from pathlib import Path

import astropy.units as u
import numpy as np
import pyproj
from astropy.table import QTable

from simtools import db_handler, io_handler
from simtools.layout.telescope_position import TelescopePosition
from simtools.util import names
from simtools.util.general import collect_data_from_yaml_or_dict
from simtools.util.names import lst

__all__ = ["InvalidTelescopeListFile", "LayoutArray"]


class InvalidTelescopeListFile(Exception):
    """Exception for invalid telescope list file."""


class LayoutArray:
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
        mongo_db_config,
        site,
        label=None,
        name=None,
        layout_center_data=None,
        corsika_telescope_data=None,
        telescope_list_file=None,
    ):
        """
        Initialize LayoutArray.
        """

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init LayoutArray")

        self.mongo_db_config = mongo_db_config
        self.label = label
        self.name = name
        self.site = names.validate_site_name(site)
        self.io_handler = io_handler.IOHandler()

        self._telescope_list = []
        self._epsg = None
        if telescope_list_file is None:
            self._initialize_coordinate_systems(layout_center_data)
            self._initialize_corsika_telescope(corsika_telescope_data)
        else:
            self.read_telescope_list_file(telescope_list_file)

    @classmethod
    def from_layout_array_name(cls, mongo_db_config, layout_array_name, label=None):
        """
        Read telescope list from file for given layout name (e.g. South-4LST, North-Prod5, ...).
        Layout definitions are given in the `data/layout` path.

        Parameters
        ----------
        mongo_db_config: dict
            MongoDB configuration.
        layout_array_name: str
            e.g. South-4LST, North-Prod5 ...
        label: str
            Instance label. Important for output file naming.

        Returns
        -------
        LayoutArray
            Instance of the LayoutArray.
        """

        split_name = layout_array_name.split("-")
        site_name = names.validate_site_name(split_name[0])
        array_name = names.validate_layout_array_name(split_name[1])
        valid_layout_array_name = site_name + "-" + array_name

        layout = cls(
            site=site_name,
            mongo_db_config=mongo_db_config,
            name=valid_layout_array_name,
            label=label,
        )

        telescope_list_file = layout.io_handler.get_input_data_file(
            "layout", f"telescope_positions-{valid_layout_array_name}.ecsv"
        )
        layout.read_telescope_list_file(telescope_list_file)

        return layout

    def __len__(self):
        return len(self._telescope_list)

    def __getitem__(self, i):
        return self._telescope_list[i]

    def _initialize_corsika_telescope(self, corsika_dict=None):
        """
        Initialize Dictionary for CORSIKA telescope parameters. Allow input from different sources\
        (dictionary, yaml, ecsv header), which require checks to handle units correctly.

        Parameters
        ----------
        corsika_dict dict
            dictionary with CORSIKA telescope parameters

        """
        self._corsika_telescope = {}

        if corsika_dict is not None:
            self._logger.debug(f"Initialize CORSIKA telescope parameters from dict: {corsika_dict}")
        else:
            self._logger.debug("Initialize CORSIKA telescope parameters from file")
            corsika_dict = self._from_corsika_file_to_dict()

        self._initialize_corsika_telescope_from_dict(corsika_dict)

    def _from_corsika_file_to_dict(self, file_name=None):
        """
        Get the corsika parameter file and return a dictionary with the keys necessary to\
        initialize this class.

        Parameters
        ----------
        file_name: str or Path
            File from which to extract the corsika parameters. Default is \
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
            corsika_parameters_dict = collect_data_from_yaml_or_dict(
                self.io_handler.get_input_data_file("parameters", "corsika_parameters.yml"), None
            )
        else:
            if not isinstance(file_name, Path):
                file_name = Path(file_name)
            if file_name.exists():
                corsika_parameters_dict = collect_data_from_yaml_or_dict(file_name, None)
            else:
                raise FileNotFoundError

        corsika_dict = {}
        corsika_pars = ["corsika_sphere_radius", "corsika_sphere_center"]
        for simtools_par in corsika_pars:
            corsika_par = names.translate_simtools_to_corsika(simtools_par)
            corsika_dict[simtools_par] = {}
            for tel_type in names.all_telescope_class_names:
                unit = corsika_parameters_dict[corsika_par][tel_type]["unit"]
                corsika_dict[simtools_par][tel_type] = corsika_parameters_dict[corsika_par][
                    tel_type
                ]["value"] * u.Unit(unit)

        db = db_handler.DatabaseHandler(mongo_db_config=self.mongo_db_config)
        self._logger.debug("Reading site parameters from DB")
        _site_pars = db.get_site_parameters(self.site, "Current", only_applicable=True)
        corsika_dict["corsika_obs_level"] = _site_pars["altitude"]["Value"] * u.Unit(
            _site_pars["altitude"]["units"]
        )

        return corsika_dict

    @staticmethod
    def _initialize_sphere_parameters(sphere_dict):
        """
        Set CORSIKA sphere parameters from dictionary. Type of input varies and depend on data \
        source for these parameters.

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
        except (TypeError, KeyError):
            pass

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
        try:
            self._corsika_telescope["corsika_sphere_center"] = self._initialize_sphere_parameters(
                corsika_dict["corsika_sphere_center"]
            )
        except (TypeError, KeyError):
            pass
        try:
            self._corsika_telescope["corsika_sphere_radius"] = self._initialize_sphere_parameters(
                corsika_dict["corsika_sphere_radius"]
            )
        except (TypeError, KeyError):
            pass

    def _initialize_coordinate_systems(self, center_dict=None):
        """
        Initialize array center and coordinate systems.

        Parameters
        ----------
        center_dict: dict
            dictionary with coordinates of array center.

        Raises
        ------
        TypeError
            invalid array center definition.

        """
        self._logger.debug(f"Initialize array center coordinate systems: {center_dict}")

        self._array_center = TelescopePosition()
        self._array_center.name = "array_center"
        self._array_center.set_coordinates("corsika", 0.0 * u.m, 0.0 * u.m, 0.0 * u.m)

        center_dict = {} if center_dict is None else center_dict
        try:
            self._array_center.set_coordinates(
                "mercator",
                u.Quantity(center_dict.get("center_lat", np.nan * u.deg)),
                u.Quantity(center_dict.get("center_lon", np.nan * u.deg)),
            )
        except TypeError:
            pass
        try:
            self._epsg = center_dict.get("EPSG", None)
            self._array_center.set_coordinates(
                "utm",
                u.Quantity(center_dict.get("center_easting", np.nan * u.m)),
                u.Quantity(center_dict.get("center_northing", np.nan * u.m)),
            )
        except TypeError:
            pass
        try:
            self._array_center.set_altitude(u.Quantity(center_dict.get("center_alt", 0.0 * u.m)))
        except TypeError:
            pass
        try:
            _name = center_dict.get("array_name")
            self.name = _name if _name is not None else self.name
        except KeyError:
            pass

        self._array_center.convert_all(
            crs_local=self._get_crs_local(),
            crs_wgs84=self._get_crs_wgs84(),
            crs_utm=self._get_crs_utm(),
        )

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

        """

        if names.get_telescope_type(tel_name) is not None:
            return self._corsika_telescope["corsika_sphere_center"][
                names.get_telescope_type(tel_name)
            ]

        return 0.0 * u.m

    def _load_telescope_names(self, row):
        """
        Read and set telescope names

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
                tel.asset_code = names.get_telescope_type(tel.name)
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

            try:
                if not isinstance(row["pos_x"], u.Quantity) and not isinstance(
                    row["pos_y"], u.Quantity
                ):
                    tel.set_coordinates(
                        "corsika",
                        row["pos_x"] * table["pos_x"].unit,
                        row["pos_y"] * table["pos_y"].unit,
                    )
                else:
                    tel.set_coordinates("corsika", row["pos_x"], row["pos_y"])
            except KeyError:
                pass
            try:
                if not isinstance(row["utm_east"], u.Quantity) and not isinstance(
                    row["utm_north"], u.Quantity
                ):
                    tel.set_coordinates(
                        "utm",
                        row["utm_east"] * table["utm_east"].unit,
                        row["utm_north"] * table["utm_north"].unit,
                    )
                else:
                    tel.set_coordinates("utm", row["utm_east"], row["utm_north"])
            except KeyError:
                pass
            try:
                if not isinstance(row["lat"], u.Quantity) and not isinstance(
                    row["lon"], u.Quantity
                ):
                    tel.set_coordinates(
                        "mercator", row["lat"] * table["lat"].unit, row["lon"] * table["lon"].unit
                    )
                else:
                    tel.set_coordinates("mercator", row["lat"], row["lon"])
            except KeyError:
                pass
            try:
                if not isinstance(row["pos_z"], u.Quantity):
                    tel.set_altitude(
                        self._altitude_from_corsika_z(
                            pos_z=row["pos_z"] * table["pos_z"].unit, tel_name=tel.name
                        )
                    )
                else:
                    tel.set_altitude(
                        self._altitude_from_corsika_z(pos_z=row["pos_z"], tel_name=tel.name)
                    )
            except KeyError:
                pass
            try:
                if not isinstance(row["alt"], u.Quantity):
                    tel.set_altitude(row["alt"] * table["alt"].unit)
                else:
                    tel.set_altitude(row["alt"])
            except KeyError:
                pass
            self._telescope_list.append(tel)

    def read_telescope_list_file(self, telescope_list_file):
        """
        Read list of telescopes from a ecsv file.

        Parameters
        ----------
        telescope_list_file: str or Path
            Path to the telescope list file.

        Returns
        -------
        dict
            Dictionary with the telescope layout information.

        Raises
        ------
        FileNotFoundError
            If file cannot be opened.

        """
        try:
            table = QTable.read(telescope_list_file, format="ascii.ecsv")
        except FileNotFoundError:
            self._logger.error(f"Error reading list of array elements from {telescope_list_file}")
            raise

        self._logger.info(f"Reading array elements from {telescope_list_file}")

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
        File metadata for export of array element list to file. Included array center definiton,\
        CORSIKA telescope parameters, and EPSG centre

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

    def _set_telescope_list_file(self, crs_name):
        """
        Set file location for writing of telescope list

        Parameters
        ----------
        crs_name: str
            Name of coordinate system to be used for export.

        Returns
        -------
        Path
            Output file

        """

        _output_directory = self.io_handler.get_output_directory(self.label, "layout")

        _name = crs_name if self.name is None else self.name + "-" + crs_name
        self.telescope_list_file = _output_directory.joinpath(
            names.layout_telescope_list_file_name(_name, None)
        )

    def export_telescope_list(self, crs_name, corsika_z=False):
        """
        Export array elements positions to ECSV file

        Parameters
        ----------
        crs_name: str
            Name of coordinate system to be used for export.
        corsika_z: bool
            Write telescope height in CORSIKA coordinates (for CORSIKA system).
        """

        table = QTable(meta=self._get_export_metadata(crs_name == "corsika"))

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
                table["pos_z"] = pos_z
            else:
                table[_name_z] = pos_z
        except IndexError:
            pass

        self._set_telescope_list_file(crs_name)
        self._logger.info(f"Exporting telescope list to {self.telescope_list_file}")
        table.write(self.telescope_list_file, format="ascii.ecsv", overwrite=True)

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
            pos_x, pos_y, pos_z = tel.get_coordinates("corsika")
            try:
                sphere_radius = self._corsika_telescope["corsika_sphere_radius"][
                    names.get_telescope_type(tel.name)
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
            corsika_list += f"\t {pos_x.value:.3f}E2"
            corsika_list += f"\t {pos_y.value:.3f}E2"
            corsika_list += f"\t {pos_z.value:.3f}E2"
            corsika_list += f"\t {sphere_radius.value:.3f}E2"
            corsika_list += f"\t # {tel.name}\n"

        return corsika_list

    def _print_all(self):
        """ "
        Print all columns for all coordinate systems.

        """

        print(f"LayoutArray: {self.name}")
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
        Print list of telescopes in current layout.

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

        wgs84 = self._get_crs_wgs84()
        crs_local = self._get_crs_local()
        crs_utm = self._get_crs_utm()

        for tel in self._telescope_list:
            tel.convert_all(
                crs_local=crs_local,
                crs_wgs84=wgs84,
                crs_utm=crs_utm,
            )

    def _get_crs_local(self):
        """
        Local coordinate system definition.

        Returns
        -------
        pyproj.CRS
            local coordinate system.

        """
        if self._array_center:
            _center_lat, _center_lon, _ = self._array_center.get_coordinates("mercator")
            if not np.isnan(_center_lat.value) and not np.isnan(_center_lon.value):
                proj4_string = (
                    "+proj=tmerc +ellps=WGS84 +datum=WGS84"
                    + f" +lon_0={_center_lon} +lat_0={_center_lat}"
                    + " +axis=nwu +units=m +k_0=1.0"
                )
                crs_local = pyproj.CRS.from_proj4(proj4_string)
                self._logger.debug(f"Local Mercator projection: {crs_local}")
                return crs_local

        self._logger.debug("crs_local cannot be built: missing array center lon and lat")

        return None

    def _get_crs_utm(self):
        """
        UTM coordinate system definition.

        Returns
        -------
        pyproj.CRS
            UTM coordinate system.

        """
        if self._epsg:
            crs_utm = pyproj.CRS.from_user_input(self._epsg)
            self._logger.debug(f"UTM system: {crs_utm}")
            return crs_utm

        self._logger.debug("crs_utm cannot be built because EPSG definition is missing")

        return None

    @staticmethod
    def _get_crs_wgs84():
        """
        WGS coordinate system definition.

        Returns
        -------
        pyproj.CRS
            WGS coordinate system.

        """
        return pyproj.CRS("EPSG:4326")

    @staticmethod
    def get_telescope_type(telescope_name):
        """
        Guess telescope type from name, e.g. "LST", "MST", "SST", "SCT".

        Parameters
        ----------
        telescope_name: str
            Telescope name

        Returns
        -------
        str
            Telescope type.
        """

        _class, _ = names.split_telescope_model_name(telescope_name)
        try:
            if _class[0:3] in (names.all_telescope_class_names):
                return _class[0:3]

        except IndexError:
            pass

    @staticmethod
    def include_radius_into_telescope_table(telescope_table):
        """
        Include the radius of the telescopes types into the astropy.table.QTable telescopes_table

        Parameters
        ----------
        file_name: str
            Name of the ecsv file with telescope layout.

        Returns
        -------
        dict
            Dictionary with the telescope layout information.
        """

        telescope_table["radius"] = [
            float(telescope_table.meta["corsika_sphere_radius"][tel_name_now[:3]].split()[0])
            for tel_name_now in telescope_table["telescope_name"]
        ]
        telescope_table["radius"].unit = u.Unit(
            telescope_table.meta["corsika_sphere_radius"][lst].split()[1]
        )
        return telescope_table
