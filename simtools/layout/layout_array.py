import logging

import astropy.units as u
import pyproj
from astropy.table import Table

import simtools.config as cfg
import simtools.io_handler as io
from simtools.layout.telescope_position import TelescopePosition
from simtools.util import names


class InvalidTelescopeListFile(Exception):
    pass


class LayoutArray:
    """
    Manage telescope positions at the array layout level.

    Methods
    -------
    fromLayoutArrayName(layoutArrayName, label=None)
        Create a LayoutArray from a layout name (e.g. South-4LST, North-Prod5, ...)
    readTelescopeListFile(telescopeListFile)
        Read list of telescopes from a ecsv file.
    addTelescope(
        telescopeName,
        posX=None,
        posY=None,
        posZ=None,
        longitude=None,
        latitude=None,
        utmEast=None,
        utmNorth=None,
        altitude=None
    )
        Add an individual telescope to the telescope list.
    exportTelescopeList(filesLocation)
        Export a ECSV file with the telescope positions.
    getNumberOfTelescopes()
        Return the number of telescopes in the list.
    getCorsikaInputList()
        Get a string with the piece of text to be added to
        the CORSIKA input file.
    printTelescopeList()
        Print list of telescopes in current layout for inspection.
    convertCoordinates()
        Perform all the possible conversions the coordinates of the tel positions.
    """

    def __init__(
        self,
        label=None,
        name=None,
        layoutCenterData=None,
        corsikaTelescopeData=None,
    ):
        """
        LayoutArray init.

        Parameters
        ----------
        name: str
            Name of the telescope (e.g L-01, S-05, ...)
        label: str
            Instance label.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be
            taken from the config.yml file.
        layoutCenterData: dict
            Dict describing array center coordinates.
        corsikaTelescopeData: dict
            Dict describing CORSIKA telescope parameters.

        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init LayoutArray")

        self.label = label

        self.name = name
        self._telescopeList = []
        self._epsg = None
        if layoutCenterData:
            self._initalizeCoordinateSystems(layoutCenterData)
        else:
            self._initalizeCoordinateSystems(self._layoutCenterDefaults())
        if corsikaTelescopeData:
            self._initializeCorsikaTelescope(corsikaTelescopeData)
        else:
            self._corsikaTelescope = {}
            self._initializeCorsikaTelescope(self._corsikaTelescopeDefault())

    @classmethod
    def fromLayoutArrayName(cls, layoutArrayName, label=None):
        """
        Create a LayoutArray from a layout name (e.g. South-4LST, North-Prod5, ...)

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
        siteName = names.validateSiteName(spl[0])
        arrayName = names.validateLayoutArrayName(spl[1])
        validLayoutArrayName = siteName + "-" + arrayName

        layout = cls(name=validLayoutArrayName, label=label)

        telescopeListFile = io.getDataFile(
            "layout", "telescope_positions-{}.ecsv".format(validLayoutArrayName)
        )
        layout.readTelescopeListFile(telescopeListFile)

        return layout

    def __len__(self):
        return len(self._telescopeList)

    def __getitem__(self, i):
        return self._telescopeList[i]

    def _initializeCorsikaTelescope(self, corsika_dict):
        """
        Initialize CORSIKA telescope parameters.

        Parameters
        ----------
        corsika_dict dict
            dictionary with CORSIKA telescope parameters

        """
        try:
            if corsika_dict["corsika_obs_level"]:
                self._corsikaTelescope["corsika_obs_level"] = u.Quantity(
                    corsika_dict["corsika_obs_level"]
                )
            else:
                self._corsikaTelescope["corsika_obs_level"] = None
            self._corsikaTelescope["corsika_sphere_center"] = {}
            for key, value in corsika_dict["corsika_sphere_center"].items():
                self._corsikaTelescope["corsika_sphere_center"][key] = u.Quantity(value)
            self._corsikaTelescope["corsika_sphere_radius"] = {}
            for key, value in corsika_dict["corsika_sphere_radius"].items():
                self._corsikaTelescope["corsika_sphere_radius"][key] = u.Quantity(value)
        except KeyError:
            pass

    def _initalizeCoordinateSystems(self, center_dict):
        """
        Initialize array center and coordinate systems

        Parameters
        ----------
        center_dict dict
            dictionary with coordinates of array center

        """

        self._arrayCenter = TelescopePosition()
        self._arrayCenter.name = "array_center"
        self._epsg = center_dict["EPSG"]

        self._arrayCenter.setLocalCoordinates(0 * u.m, 0 * u.m, 0 * u.m)
        try:
            self._arrayCenter.setMercatorCoordinates(
                u.Quantity(center_dict["center_lat"]),
                u.Quantity(center_dict["center_lon"]),
            )
            self._arrayCenter.setUtmCoordinates(
                u.Quantity(center_dict["center_easting"]),
                u.Quantity(center_dict["center_northing"]),
            )
            self._arrayCenter.setAltitude(u.Quantity(center_dict["center_alt"]))
        except TypeError:
            pass

        self._arrayCenter.convertAll(
            crsLocal=self._getCrsLocal(),
            wgs84=self._getCrsWgs84(),
            crsUtm=self._getCrsUtm(),
        )

    def _loadTelescopeList(self, table):
        """
        Load list of telescope from an astropy table

        Parameters
        ----------
        table: astropy.table
            data table with array element coordinates

        """

        for row in table:
            tel = TelescopePosition()
            try:
                tel.name = row["telescope_name"]
            except KeyError:
                msg = "Missing required row with telescope_name"
                self._logger.error(msg)
                raise InvalidTelescopeListFile(msg)

            try:
                tel.name = row["telescope_name"]
                tel.setLocalCoordinates(
                    posX=row["pos_x"] * table["pos_x"].unit,
                    posY=row["pos_y"] * table["pos_y"].unit,
                    posZ=row["pos_z"] * table["pos_z"].unit,
                )
                tel.setUtmCoordinates(
                    utmEast=row["utm_east"] * table["utm_east"].unit,
                    utmNorth=row["utm_north"] * table["utm_north"].unit,
                )
                tel.setMercatorCoordinates(
                    latitude=row["lat"] * table["lat"].unit,
                    longitude=row["lon"] * table["lon"].unit,
                )
                tel.setAltitude(altitude=row["alt"] * table["alt"].unit)
            except KeyError:
                pass

            self._telescopeList.append(tel)

    def readTelescopeListFile(self, telescopeListFile):
        """
        Read list of telescopes from a ecsv file.

        Parameters
        ----------
        telescopeListFile: str (or Path)
            Path to the telescope list file.

        Raises
        ------
        InvalidTelescopeListFile
            If cannot read telescope list file or the table does not contain
            telescope_name key.

        """
        try:
            table = Table.read(telescopeListFile, format="ascii.ecsv")
        except FileNotFoundError:
            logging.error("Error reading list of array elements from {}".format(telescopeListFile))
            raise

        self._logger.info("Reading array elements from {}".format(telescopeListFile))

        self._initializeCorsikaTelescope(table.meta)
        self._initalizeCoordinateSystems(table.meta)
        self._loadTelescopeList(table)

    @u.quantity_input(
        posX=u.m,
        posY=u.m,
        posZ=u.m,
        longitude=u.deg,
        latitude=u.deg,
        utmEast=u.m,
        utmNorth=u.m,
        altitude=u.m,
    )
    def addTelescope(
        self,
        telescopeName,
        posX=None,
        posY=None,
        posZ=None,
        longitude=None,
        latitude=None,
        utmEast=None,
        utmNorth=None,
        altitude=None,
    ):
        """
        Add an individual telescope to the telescope list.

        Parameters
        ----------
        telescopeName: str
            Name of the telescope starting with L, M or S (e.g. L-01, M-06 ...)
        posX: astropy.units.quantity.Quantity
            X coordinate in equivalent units of u.m.
        posY: astropy.units.quantity.Quantity
            Y coordinate in equivalent units of u.m.
        posZ: astropy.units.quantity.Quantity
            Z coordinate in equivalent units of u.m.
        longitude: astropy.units.quantity.Quantity
            Longitude coordinate in equivalent units of u.deg.
        latitude: astropy.units.quantity.Quantity
            Latitude coordinate in equivalent units of u.deg.
        utmEast: astropy.units.quantity.Quantity
            UTM east coordinate in equivalent units of u.deg.
        utmNorth: astropy.units.quantity.Quantity
            UTM north coordinate in equivalent units of u.deg.
        altitude: astropy.units.quantity.Quantity
            Altitude coordinate in equivalent units of u.m.
        """

        configData = {
            "posX": posX,
            "posY": posY,
            "posZ": posZ,
            "longitude": longitude,
            "latitude": latitude,
            "utmEast": utmEast,
            "utmNorth": utmNorth,
            "altitude": altitude,
        }

        tel = TelescopePosition(name=telescopeName, configData=configData)
        self._telescopeList.append(tel)

    def _get_export_metadata(self):
        """
        File metadata for export of array element list to file.
        Included array center definiton, CORSIKA telescope parameters,
        and EPSG centre

        Returns
        -------
        dict
            metadata header for array element list export

        """

        _meta = {}

        _lat = None
        _lon = None
        _east = None
        _north = None
        if self._arrayCenter:
            _lat, _lon = self._arrayCenter.getMercatorCoordinates()
            _north, _east = self._arrayCenter.getUtmCoordinates()

        _meta["center_lon"] = _lon
        _meta["center_lat"] = _lat
        _meta["center_northing"] = _north
        _meta["center_easting"] = _east
        _meta.update(self._corsikaTelescope)
        _meta["EPSG"] = self._epsg

        return _meta

    def exportTelescopeList(self, filesLocation=None):
        """
        Export array elements positions to ECSV file

        Parameters
        ----------
        filesLocation: str (or Path), optional
            Directory for output. If not given, taken from config file.

        """

        fileName = names.layoutTelescopeListFileName(self.name, None)

        self._filesLocation = cfg.getConfigArg("outputLocation", filesLocation)
        self._outputDirectory = io.getLayoutOutputDirectory(self._filesLocation, self.label)
        self._outputDirectory.mkdir(parents=True, exist_ok=True)
        self.telescopeListFile = self._outputDirectory.joinpath(fileName)

        self._logger.debug(
            "Exporting telescope list to ECSV file {}".format(self.telescopeListFile)
        )

        metaData = self._get_export_metadata()

        table = Table(meta=metaData)

        tel_names = list()
        pos_x, pos_y, pos_z = list(), list(), list()
        utm_east, utm_north = list(), list()
        longitude, latitude = list(), list()
        altitude = list()
        for tel in self._telescopeList:
            tel_names.append(tel.name)

            if tel.hasLocalCoordinates():
                x, y, z = tel.getLocalCoordinates()
                pos_x.append(x)
                pos_y.append(y)
                pos_z.append(z)

            if tel.hasMercatorCoordinates():
                lat, lon = tel.getMercatorCoordinates()
                latitude.append(lat)
                longitude.append(lon)

            if tel.hasUtmCoordinates():
                un, ue = tel.getUtmCoordinates()
                utm_east.append(ue)
                utm_north.append(un)

            if tel.hasAltitude():
                alt = tel.getAltitude()
                altitude.append(alt)

        table["telescope_name"] = tel_names

        if len(pos_x) > 0:
            table["pos_x"] = pos_x * u.m
            table["pos_y"] = pos_x * u.m
            table["pos_z"] = pos_x * u.m

        if len(latitude) > 0:
            table["lat"] = latitude * u.deg
            table["lon"] = longitude * u.deg

        if len(utm_east) > 0:
            table["utm_east"] = utm_east * u.m
            table["utm_north"] = utm_north * u.m

        if len(altitude) > 0:
            table["alt"] = altitude * u.m

        table.write(self.telescopeListFile, format="ascii.ecsv", overwrite=True)

    def getNumberOfTelescopes(self):
        """
        Return the number of telescopes in the list.

        Returns
        -------
        int
            Number of telescopes.
        """
        return len(self._telescopeList)

    def getCorsikaInputList(self):
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
            posX, posY, posZ = tel.getLocalCoordinates()
            try:
                sphereRadius = self._corsikaTelescope["corsika_sphere_radius"][
                    tel.getTelescopeSize()
                ]
            except KeyError:
                self._logger.error("Missing definition of CORSIKA sphere radius")
                raise

            corsikaList += "TELESCOPE"
            corsikaList += "\t {:.3f}E2".format(posX.value)
            corsikaList += "\t {:.3f}E2".format(posY.value)
            corsikaList += "\t {:.3f}E2".format(posZ.value)
            corsikaList += "\t {:.3f}E2".format(sphereRadius)
            corsikaList += "\t # {}\n".format(tel.name)

        return corsikaList

    def printTelescopeList(self):
        """
        Print list of telescopes in current layout for inspection.

        """

        print("LayoutArray: {}".format(self.name))
        print("ArrayCenter")
        print(self._arrayCenter)
        print("Telescopes")
        for tel in self._telescopeList:
            print(tel)

    def convertCoordinates(self):
        """Perform all the possible conversions the coordinates of the tel positions."""

        self._logger.info("Converting telescope coordinates")

        wgs84 = self._getCrsWgs84()
        crs_local = self._getCrsLocal()
        crs_utm = self._getCrsUtm()

        for tel in self._telescopeList:
            tel.convertAll(
                crsLocal=crs_local,
                wgs84=wgs84,
                crsUtm=crs_utm,
                corsikaObsLevel=self._corsikaTelescope["corsika_obs_level"],
                corsikaSphereCenter=self._corsikaTelescope["corsika_sphere_center"][
                    tel.getTelescopeSize()
                ],
            )

    def _getCrsLocal(self):
        """
        Local coordinate system definition

        Returns
        -------
        pyproj.CRS
            local coordinate system

        """
        if self._arrayCenter:
            _centerLat, _centerLong = self._arrayCenter.getMercatorCoordinates()
            if _centerLong and _centerLat:
                proj4_string = (
                    "+proj=tmerc +ellps=WGS84 +datum=WGS84"
                    + " +lon_0={} +lat_0={}".format(_centerLong, _centerLat)
                    + " +axis=nwu +units=m +k_0=1.0"
                )
                crs_local = pyproj.CRS.from_proj4(proj4_string)
                self._logger.debug("Local Mercator projection: {}".format(crs_local))
                return crs_local

        self._logger.debug("crs_local cannot be built: missing array center lon and lat")

    def _getCrsUtm(self):
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
        else:
            self._logger.debug("crs_utm cannot be built because EPSG definition is missing")

    @staticmethod
    def _getCrsWgs84():
        """
        WGS coordinate system definition

        Returns
        -------
        pyproj.CRS
            WGS coordinate system

        """
        return pyproj.CRS("EPSG:4326")

    @staticmethod
    def _layoutCenterDefaults():
        """
        Default values for array center dict

        Returns
        -------
        dict
            array center default values

        """
        return {
            "EPSG": None,
            "center_lon": None,
            "center_lat": None,
            "center_northing": None,
            "center_easting": None,
            "center_alt": None,
        }

    @staticmethod
    def _corsikaTelescopeDefault():
        """
        Default values for CORSIKA telescope dict

        Returns
        -------
        dict
            corsika telescope default values

        TODO: discuss hardwired corsika parameters

        """
        return {
            "corsika_obs_level": None,
            "corsika_sphere_center": {
                "LST": 16.0 * u.m,
                "MST": 9.0 * u.m,
                "SST": 3.25 * u.m,
            },
            "corsika_sphere_radius": {
                "LST": 12.5 * u.m,
                "MST": 9.6 * u.m,
                "SST": 3.5 * u.m,
            },
        }
