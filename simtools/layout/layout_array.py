import logging

import astropy.units as u
import numpy as np
import pyproj
from astropy.table import Table

from simtools import io_handler
from simtools.layout.telescope_position import TelescopePosition
from simtools.util import names
from simtools.util.general import collect_data_from_yaml_or_dict


class InvalidTelescopeListFile(Exception):
    pass


class LayoutArray:
    """
    Manage telescope positions at the array layout level.

    Methods
    -------
    from_layout_array_name(layoutArrayName, label=None)
        Create a LayoutArray from a layout name (e.g. South-4LST, North-Prod5, ...)
    read_telescope_list_file(telescopeListFile)
        Read list of telescopes from a ecsv file.
    add_telescope(
        telescopeName,
        crsName,
        xx,
        yy,
        altitude=None
        telCorsikaZ=None
    )
        Add an individual telescope to the telescope list.
    export_telescope_list()
        Export a ECSV file with the telescope positions.
    get_number_of_telescopes()
        Return the number of telescopes in the list.
    get_corsika_input_list()
        Get a string with the piece of text to be added to
        the CORSIKA input file.
    print_telescope_list()
        Print list of telescopes in current layout for inspection.
    convert_coordinates()
        Perform all the possible conversions the coordinates of the tel positions.
    """

    def __init__(
        self,
        label=None,
        name=None,
        layoutCenterData=None,
        corsikaTelescopeData=None,
        telescopeListFile=None,
    ):
        """
        LayoutArray init.

        Parameters
        ----------
        name: str
            Name of the layout.
        label: str
            Instance label.
        layoutCenterData: dict
            Dict describing array center coordinates.
        corsikaTelescopeData: dict
            Dict describing CORSIKA telescope parameters.
        telescopeListFile: str (or Path)
            Path to the telescope list file.

        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init LayoutArray")

        self.label = label
        self.name = name
        self.io_handler = io_handler.IOHandler()

        self._telescopeList = []
        self._epsg = None
        if telescopeListFile is None:
            self._initialize_coordinate_systems(layoutCenterData)
            self._initialize_corsika_telescope(corsikaTelescopeData)
        else:
            self.read_telescope_list_file(telescopeListFile)

    @classmethod
    def from_layout_array_name(cls, layoutArrayName, label=None):
        """
        Read telescope list from file for given layout name (e.g. South-4LST, North-Prod5, ...).
        Layout definitions are given in the `data/layout` path.

        Parameters
        ----------
        layoutArrayName: str
            e.g. South-4LST, North-Prod5 ...
        label: str, optional
            Instance label. Important for output file naming.

        Returns
        -------
        Instance of the LayoutArray class.
        """

        spl = layoutArrayName.split("-")
        siteName = names.validate_site_name(spl[0])
        arrayName = names.validate_layout_array_name(spl[1])
        validLayoutArrayName = siteName + "-" + arrayName

        layout = cls(name=validLayoutArrayName, label=label)

        telescopeListFile = layout.io_handler.get_input_data_file(
            "layout", "telescope_positions-{}.ecsv".format(validLayoutArrayName)
        )
        layout.read_telescope_list_file(telescopeListFile)

        return layout

    def __len__(self):
        return len(self._telescopeList)

    def __getitem__(self, i):
        return self._telescopeList[i]

    def _initialize_corsika_telescope(self, corsikaDict=None):
        """
        Initialize Dictionary for CORSIKA telescope parameters.
        Allow input from different sources (dictionary, yaml, ecsv header), which
        require checks to handle units correctly.

        Parameters
        ----------
        corsikaDict dict
            dictionary with CORSIKA telescope parameters

        """
        self._corsikaTelescope = {}

        if corsikaDict is not None:
            self._logger.debug(
                "Initialize CORSIKA telescope parameters from dict: {}".format(corsikaDict)
            )
            self._initialize_corsika_telescope_from_dict(corsikaDict)
        else:
            self._logger.debug("Initialize CORSIKA telescope parameters from file")
            self._initialize_corsika_telescope_from_dict(
                collect_data_from_yaml_or_dict(
                    self.io_handler.get_input_data_file("corsika", "corsika_parameters.yml"), None
                )
            )

    @staticmethod
    def _initialize_sphere_parameters(sphere_dict):
        """
        Set CORSIKA sphere parameters from dictionary.
        Type of input varies and depend on data source for these parameters.

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
                if isinstance(value, u.Quantity) or isinstance(value, str):
                    _sphere_dict_cleaned[key] = u.Quantity(value)
                else:
                    _sphere_dict_cleaned[key] = value["value"] * u.Unit(value["unit"])
        except (TypeError, KeyError):
            pass

        return _sphere_dict_cleaned

    def _initialize_corsika_telescope_from_dict(self, corsikaDict):
        """
        Initialize CORSIKA telescope parameters from a dictionary.

        Parameters
        ----------
        corsikaDict dict
            dictionary with CORSIKA telescope parameters

        """

        try:
            self._corsikaTelescope["corsika_obs_level"] = u.Quantity(
                corsikaDict["corsika_obs_level"]
            )
        except (TypeError, KeyError):
            self._corsikaTelescope["corsika_obs_level"] = np.nan * u.m
        try:
            self._corsikaTelescope["corsika_sphere_center"] = self._initialize_sphere_parameters(
                corsikaDict["corsika_sphere_center"]
            )
        except (TypeError, KeyError):
            pass
        try:
            self._corsikaTelescope["corsika_sphere_radius"] = self._initialize_sphere_parameters(
                corsikaDict["corsika_sphere_radius"]
            )
        except (TypeError, KeyError):
            pass

    def _initialize_coordinate_systems(self, center_dict=None, defaults_init=False):
        """
        Initialize array center and coordinate systems.

        Parameters
        ----------
        center_dict: dict
            dictionary with coordinates of array center.
        defaults_init: bool
            default initialisation without transformation in all available projections.

        Raises
        ------
        TypeError
            invalid array center definition

        """
        self._logger.debug("Initialize array center coordinate systems: {}".format(center_dict))

        self._arrayCenter = TelescopePosition()
        self._arrayCenter.name = "array_center"
        self._arrayCenter.set_coordinates("corsika", 0.0 * u.m, 0.0 * u.m, 0.0 * u.m)

        center_dict = {} if center_dict is None else center_dict
        try:
            self._arrayCenter.set_coordinates(
                "mercator",
                u.Quantity(center_dict.get("center_lat", np.nan * u.deg)),
                u.Quantity(center_dict.get("center_lon", np.nan * u.deg)),
            )
        except TypeError:
            pass
        try:
            self._epsg = center_dict.get("EPSG", None)
            self._arrayCenter.set_coordinates(
                "utm",
                u.Quantity(center_dict.get("center_easting", np.nan * u.m)),
                u.Quantity(center_dict.get("center_northing", np.nan * u.m)),
            )
        except TypeError:
            pass
        try:
            self._arrayCenter.set_altitude(u.Quantity(center_dict.get("center_alt", 0.0 * u.m)))
        except TypeError:
            pass
        try:
            _name = center_dict.get("array_name")
            self.name = _name if _name is not None else self.name
        except KeyError:
            pass

        self._arrayCenter.convert_all(
            crsLocal=self._get_crs_local(),
            crsWgs84=self._get_crs_wgs84(),
            crsUtm=self._get_crs_utm(),
        )

    def _altitude_from_corsika_z(self, pos_z=None, altitude=None, tel_name=None):
        """
        Calculate altitude from CORSIKA z-coordinate (if pos_z is given) or
        CORSIKA z-coordinate from altitude (if altitude is given)

        Parameters
        ----------
        pos_z: astropy.Quantity
            CORSIKA z-coordinate of telescope in equivalent units of meter.
        altitude: astropy.Quantity
            Telescope altitude in equivalent units of meter.
        tel_name: str
            Telescope Name

        Returns
        -------
        astropy.Quantity
            Altitude or CORSIKA z-coordinate (np.nan in case of ill-defined value)

        """

        if pos_z is not None and altitude is None:
            return TelescopePosition.convert_telescope_altitude_from_corsika_system(
                pos_z,
                self._corsikaTelescope["corsika_obs_level"],
                self._get_corsika_sphere_center(tel_name),
            )
        if altitude is not None and pos_z is None:
            return TelescopePosition.convert_telescope_altitude_to_corsika_system(
                altitude,
                self._corsikaTelescope["corsika_obs_level"],
                self._get_corsika_sphere_center(tel_name),
            )

    def _get_corsika_sphere_center(self, tel_name):
        """
        Return CORSIKA sphere center value for given telescope

        Parameters
        ----------
        tel_name: str
            Telescope Name

        Returns
        -------
        astropy.Quantity
            Telescope sphere center value (0.0*u.m if sphere center is not defined)

        """

        if self.get_telescope_type(tel_name) is not None:
            return self._corsikaTelescope["corsika_sphere_center"][
                self.get_telescope_type(tel_name)
            ]

        return 0.0 * u.m

    def _load_telescope_names(self, row):
        """
        Read and set telescope names

        Parameters
        ----------
        row: astropy table row
            table row

        Returns
        -------
        tel: TelescopePosition
            telescope position

        Raises
        ------
        InvalidTelescopeListFile
            in case neither telescope name or asset_code / sequence number are given

        """

        tel = TelescopePosition()
        try:
            tel.name = row["telescope_name"]
            if "asset_code" not in row:
                tel.asset_code = self.get_telescope_type(tel.name)
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
        Load list of telescope from an astropy table

        Parameters
        ----------
        table: astropy.table
            data table with array element coordinates

        """

        for row in table:
            tel = self._load_telescope_names(row)

            try:
                tel.set_coordinates(
                    "corsika",
                    row["pos_x"] * table["pos_x"].unit,
                    row["pos_y"] * table["pos_y"].unit,
                )
            except KeyError:
                pass
            try:
                tel.set_coordinates(
                    "utm",
                    row["utm_east"] * table["utm_east"].unit,
                    row["utm_north"] * table["utm_north"].unit,
                )
            except KeyError:
                pass
            try:
                tel.set_coordinates(
                    "mercator", row["lat"] * table["lat"].unit, row["lon"] * table["lon"].unit
                )
            except KeyError:
                pass
            try:
                tel.set_altitude(
                    self._altitude_from_corsika_z(
                        pos_z=row["pos_z"] * table["pos_z"].unit, tel_name=tel.name
                    )
                )
            except KeyError:
                pass
            try:
                tel.set_altitude(row["alt"] * table["alt"].unit)
            except KeyError:
                pass

            self._telescopeList.append(tel)

    def read_telescope_list_file(self, telescopeListFile):
        """
        Read list of telescopes from a ecsv file.

        Parameters
        ----------
        telescopeListFile: str (or Path)
            Path to the telescope list file.

        Raises
        ------
        FileNotFoundError
            If file cannot be opened.

        """
        try:
            table = Table.read(telescopeListFile, format="ascii.ecsv")
        except FileNotFoundError:
            self._logger.error(
                "Error reading list of array elements from {}".format(telescopeListFile)
            )
            raise

        self._logger.info("Reading array elements from {}".format(telescopeListFile))

        self._initialize_corsika_telescope(table.meta)
        self._initialize_coordinate_systems(table.meta)
        self._load_telescope_list(table)

    def add_telescope(
        self,
        telescopeName,
        crsName,
        xx,
        yy,
        altitude=None,
        telCorsikaZ=None,
    ):
        """
        Add an individual telescope to the telescope list.

        Parameters
        ----------
        telescopeName: str
            Name of the telescope starting with L, M or S (e.g. LST-01, MST-06 ...)
        crsName: str
            Name of coordinate system
        xx: astropy.units.quantity.Quantity
            x-coordinate for the given coordinate system
        yy: astropy.units.quantity.Quantity
            y-coordinate for the given coordinate system
        altitude: astropy.units.quantity.Quantity
            Altitude coordinate in equivalent units of u.m.
        telCorsikaZ: astropy.units.quantity.Quantity
            CORSIKA z-position (requires setting of CORSIKA observation level and telescope
            sphere center).
        """

        tel = TelescopePosition(name=telescopeName)
        tel.set_coordinates(crsName, xx, yy)
        if altitude is not None:
            tel.set_altitude(altitude)
        elif telCorsikaZ is not None:
            tel.set_altitude(self._altitude_from_corsika_z(pos_z=telCorsikaZ, tel_name=tel.name))
        self._telescopeList.append(tel)

    def _get_export_metadata(self, export_corsika_meta=False):
        """
        File metadata for export of array element list to file.
        Included array center definiton, CORSIKA telescope parameters,
        and EPSG centre

        Parameters
        ----------
        export_corsika_meta: bool
            write CORSIKA metadata

        Returns
        -------
        dict
            metadata header for array element list export

        """

        _meta = {
            "center_lon": None,
            "center_lat": None,
            "center_northing": None,
            "center_easting": None,
            "center_alt": None,
        }
        if self._arrayCenter:
            _meta["center_lat"], _meta["center_lon"], _ = self._arrayCenter.get_coordinates(
                "mercator"
            )
            (
                _meta["center_easting"],
                _meta["center_northing"],
                _meta["center_alt"],
            ) = self._arrayCenter.get_coordinates("utm")
        if export_corsika_meta:
            _meta.update(self._corsikaTelescope)
        _meta["EPSG"] = self._epsg
        _meta["array_name"] = self.name

        return _meta

    def _set_telescope_list_file(self, crsName):
        """
        Set file location for writing of telescope list

        Parameters
        ----------
        crsName: str
            Name of coordinate system to be used for export.

        Returns
        -------
        Path
            Output file

        """

        _outputDirectory = self.io_handler.get_output_directory(self.label, "layout")

        _name = crsName if self.name is None else self.name + "-" + crsName
        self.telescopeListFile = _outputDirectory.joinpath(
            names.layout_telescope_list_file_name(_name, None)
        )

    def export_telescope_list(self, crsName, corsikaZ=False):
        """
        Export array elements positions to ECSV file

        Parameters
        ----------
        crsName: str
            Name of coordinate system to be used for export.
        corsikaZ: bool
            Write telescope height in CORSIKA coordinates (for CORSIKA system)

        """

        table = Table(meta=self._get_export_metadata(crsName == "corsika"))

        tel_names, asset_code, sequence_number, geo_code = list(), list(), list(), list()
        pos_x, pos_y, pos_z = list(), list(), list()
        for tel in self._telescopeList:
            tel_names.append(tel.name)
            asset_code.append(tel.asset_code)
            sequence_number.append(tel.sequence_number)
            geo_code.append(tel.geo_code)
            x, y, z = tel.get_coordinates(crsName)
            if corsikaZ:
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
            _nameX, _nameY, _nameZ = self._telescopeList[0].get_coordinates(
                crs_name=crsName, coordinate_field="name"
            )
            table[_nameX] = pos_x
            table[_nameY] = pos_y
            if corsikaZ:
                table["pos_z"] = pos_z
            else:
                table[_nameZ] = pos_z
        except IndexError:
            pass

        self._set_telescope_list_file(crsName)
        self._logger.info("Exporting telescope list to {}".format(self.telescopeListFile))
        table.write(self.telescopeListFile, format="ascii.ecsv", overwrite=True)

    def get_number_of_telescopes(self):
        """
        Return the number of telescopes in the list.

        Returns
        -------
        int
            Number of telescopes.
        """
        return len(self._telescopeList)

    def get_corsika_input_list(self):
        """
        Get a string with the piece of text to be added to
        the CORSIKA input file.

        Returns
        -------
        str
            Piece of text to be added to the CORSIKA input file.
        """
        corsikaList = ""
        for tel in self._telescopeList:
            posX, posY, posZ = tel.get_coordinates("corsika")
            try:
                sphereRadius = self._corsikaTelescope["corsika_sphere_radius"][
                    self.get_telescope_type(tel.name)
                ]
            except KeyError:
                self._logger.error("Missing definition of CORSIKA sphere radius")
                raise
            try:
                posZ = tel.convert_telescope_altitude_to_corsika_system(
                    posZ,
                    self._corsikaTelescope["corsika_obs_level"],
                    self._get_corsika_sphere_center(tel.name),
                )
            except KeyError:
                self._logger.error("Missing definition of CORSIKA sphere center / obs_level")
                raise

            corsikaList += "TELESCOPE"
            corsikaList += "\t {:.3f}E2".format(posX.value)
            corsikaList += "\t {:.3f}E2".format(posY.value)
            corsikaList += "\t {:.3f}E2".format(posZ.value)
            corsikaList += "\t {:.3f}E2".format(sphereRadius.value)
            corsikaList += "\t # {}\n".format(tel.name)

        return corsikaList

    def _print_all(self):
        """ "
        Print all columns for all coordinate systems.

        """

        print("LayoutArray: {}".format(self.name))
        print("ArrayCenter")
        print(self._arrayCenter)
        print("Telescopes")
        for tel in self._telescopeList:
            print(tel)

    def _print_compact(self, compact_printing, corsikaZ=False):
        """
        Compact printing of list of telescopes.


        Parameters
        ----------
        compact_printing: str
            Compact printout for a single coordinate system.
            Coordinates in all systems are printed, if compact_printing is None.
        corsikaZ: bool
            Print telescope height in CORSIKA coordinates (for CORSIKA system)

        """

        for tel in self._telescopeList:
            if corsikaZ:
                _corsika_obs_level = self._corsikaTelescope["corsika_obs_level"]
                _corsika_sphere_center = self._get_corsika_sphere_center(tel.name)
            else:
                _corsika_obs_level = None
                _corsika_sphere_center = None

            tel.print_compact_format(
                crs_name=compact_printing,
                print_header=(tel == self._telescopeList[0]),
                corsika_obs_level=_corsika_obs_level,
                corsika_sphere_center=_corsika_sphere_center,
            )

    def print_telescope_list(self, compact_printing="", corsikaZ=False):
        """
        Print list of telescopes in current layout.

        Parameters
        ----------
        compact_printing: str
            Compact printout for a single coordinate system
        corsikaZ: bool
            Print telescope height in CORSIKA coordinates (for CORSIKA system)

        """
        if len(compact_printing) == 0:
            self._print_all()
        else:
            self._print_compact(compact_printing, corsikaZ)

    def convert_coordinates(self):
        """Perform all the possible conversions the coordinates of the tel positions."""

        self._logger.info("Converting telescope coordinates")

        wgs84 = self._get_crs_wgs84()
        crs_local = self._get_crs_local()
        crs_utm = self._get_crs_utm()

        for tel in self._telescopeList:
            tel.convert_all(
                crsLocal=crs_local,
                crsWgs84=wgs84,
                crsUtm=crs_utm,
            )

    def _get_crs_local(self):
        """
        Local coordinate system definition

        Returns
        -------
        pyproj.CRS
            local coordinate system

        """
        if self._arrayCenter:
            _centerLat, _centerLon, _ = self._arrayCenter.get_coordinates("mercator")
            if not np.isnan(_centerLat.value) and not np.isnan(_centerLon.value):
                proj4_string = (
                    "+proj=tmerc +ellps=WGS84 +datum=WGS84"
                    + " +lon_0={} +lat_0={}".format(_centerLon, _centerLat)
                    + " +axis=nwu +units=m +k_0=1.0"
                )
                crs_local = pyproj.CRS.from_proj4(proj4_string)
                self._logger.debug("Local Mercator projection: {}".format(crs_local))
                return crs_local

        self._logger.debug("crs_local cannot be built: missing array center lon and lat")

    def _get_crs_utm(self):
        """
        UTM coordinate system definition

        Returns
        -------
        pyproj.CRS
            UTM coordinate system

        """
        if self._epsg:
            crs_utm = pyproj.CRS.from_user_input(self._epsg)
            self._logger.debug("UTM system: {}".format(crs_utm))
            return crs_utm

        self._logger.debug("crs_utm cannot be built because EPSG definition is missing")

    @staticmethod
    def _get_crs_wgs84():
        """
        WGS coordinate system definition

        Returns
        -------
        pyproj.CRS
            WGS coordinate system

        """
        return pyproj.CRS("EPSG:4326")

    @staticmethod
    def get_telescope_type(telescope_name):
        """
        Guess telescope type from name

        """

        _class, _ = names.split_telescope_model_name(telescope_name)
        try:
            if _class[0:3] in ("LST", "MST", "SST", "SCT"):
                return _class[0:3]

        except IndexError:
            pass
