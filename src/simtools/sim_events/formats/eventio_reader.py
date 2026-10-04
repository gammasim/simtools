"""Read CORSIKA IACT and sim_telarray eventio files."""

import logging
import warnings

import numpy as np
from eventio import EventIOFile, iact
from eventio.simtel import (
    ArrayEvent,
    MCEvent,
    MCRunHeader,
    MCShower,
    RunHeader,
    TrackingPosition,
    TriggerInformation,
)

from simtools.corsika.primary_particle import PrimaryParticle
from simtools.simtel.simtel_io_metadata import (
    get_sim_telarray_telescope_id_to_telescope_name_mapping,
    read_sim_telarray_metadata,
)
from simtools.utils.geometry import calculate_circular_mean, geographic_to_corsika_azimuth
from simtools.utils.names import get_common_identifier_from_array_element_name

# Suppress all UserWarnings from corsikaio - no CORSIKA versions <7.7 are supported anyway
warnings.filterwarnings("ignore", category=UserWarning, module=r"corsikaio\.subblocks\..*")


def get_corsika_run_number(file):
    """
    Return the CORSIKA run number from an eventio (CORSIKA IACT or sim_telarray) file.

    Parameters
    ----------
    file: str
        Path to the eventio file.

    Returns
    -------
    int, None
        CORSIKA run number. Returns None if not found.
    """
    run_header = get_combined_eventio_run_header(file)
    if run_header and "run" in run_header:
        return run_header["run"]
    run_header, _ = get_corsika_run_and_event_headers(file)
    try:
        return int(run_header["run_number"])
    except TypeError, KeyError, ValueError:
        return None


def get_combined_eventio_run_header(sim_telarray_file):
    """
    Return the CORSIKA run header information from an eventio (sim_telarray) file.

    Reads both RunHeader and MCRunHeader object from file and returns a merged dictionary.
    Adds primary id from the first event.

    Parameters
    ----------
    sim_telarray_file: str
        Path to the sim_telarray file.

    Returns
    -------
    dict, None
        CORSIKA run header. Returns None if not found.
    """
    run_header = mc_run_header = None
    primary_id = None

    with EventIOFile(sim_telarray_file) as f:
        for o in f:
            if isinstance(o, RunHeader) and run_header is None:
                run_header = o.parse()
            elif isinstance(o, MCRunHeader) and mc_run_header is None:
                mc_run_header = o.parse()
            elif isinstance(o, MCShower):  # get primary_id from first MCShower
                primary_id = o.parse().get("primary_id")
            if run_header and mc_run_header and primary_id is not None:
                break

    run_header = run_header or {}
    mc_run_header = mc_run_header or {}
    if primary_id is not None:
        mc_run_header["primary_id"] = primary_id
    return run_header | mc_run_header or None


def get_corsika_run_and_event_headers(corsika_iact_file):
    """
    Return the CORSIKA run and event headers from a CORSIKA IACT eventio file.

    Parameters
    ----------
    corsika_iact_file: str, Path
        Path to the CORSIKA IACT eventio file.

    Returns
    -------
    tuple
        CORSIKA run header and event header as dictionaries.
    """
    run_header = event_header = None

    with EventIOFile(corsika_iact_file) as f:
        for o in f:
            if isinstance(o, iact.RunHeader) and run_header is None:
                run_header = o.parse()
            elif isinstance(o, iact.EventHeader) and event_header is None:
                event_header = o.parse()
            if run_header and event_header:
                break

    return _header_mapping(run_header), _header_mapping(event_header)


def _header_mapping(header):
    """Expose native NumPy header records as named fields."""
    if isinstance(header, np.void):
        return {name: header[name] for name in header.dtype.names}
    return header


def _iact_geographic_azimuth(event_header):
    """Convert the CORSIKA shower frame to geographic arrival azimuth in degrees."""
    return (
        geographic_to_corsika_azimuth(np.degrees(event_header["azimuth"]))
        + np.degrees(event_header["angle_array_x_magnetic_north"])
    ) % 360


def get_simulated_events(event_io_file):
    """
    Return the number of shower and MC events from a simulation (eventio) file.

    For a sim_telarray file, the number of simulated showers and MC events is
    determined by counting the number of MCShower (type id 2020) and MCEvent
    objects (type id 2021). For a CORSIKA IACT file, the number of simulated
    showers is determined by counting the number of IACTShower (type id 1202).

    Parameters
    ----------
    event_io_file: str, Path
        Path to the eventio file.

    Returns
    -------
    tuple
        Number of showers and number of MC events (MC events for sim_telarray files only).
    """
    counts = {1202: 0, 2020: 0, 2021: 0}
    with EventIOFile(event_io_file) as f:
        for o in f:
            t = o.header.type
            if t in counts:
                counts[t] += 1
    return counts[2020] if counts[2020] else counts[1202], counts[2021]


class EventioReader:
    """Read simulation information and reduced event rows from eventio files.

    Parameters
    ----------
    file_name : str or pathlib.Path
        Simulation input file.
    """

    def __init__(self, file_name):
        self.file_name = file_name
        self._logger = logging.getLogger(__name__)
        self.n_use = None
        self.shower_data = []
        self.trigger_data = []
        self.file_info = []
        self.input_metadata = []
        self.telescope_id_to_name = {}
        self._current_run_header = {}
        self._current_mc_run_header = {}
        self._current_simtel_metadata = ({}, {})

    def read_run_number(self):
        """Return the shower run number, or None if absent."""
        return get_corsika_run_number(self.file_name)

    def read_run_headers(self):
        """Return the CORSIKA IACT run and first event headers."""
        return get_corsika_run_and_event_headers(self.file_name)

    def read_combined_run_header(self):
        """Return merged sim_telarray run information, or None."""
        return get_combined_eventio_run_header(self.file_name)

    def count_events(self):
        """Return shower and reused-event counts."""
        return get_simulated_events(self.file_name)

    def read_simulation_parameters(self):
        """Return physical shower settings in degrees and metres.

        Returns
        -------
        dict
            Settings accepted by SimulationParameters, plus observation levels
            in metres. CORSIKA header angles are converted to geographic pointing.
        """
        run, event = self.read_run_headers()
        if run is None or event is None:
            raise ValueError(f"Missing IACT run or event header in '{self.file_name}'.")
        zenith = 0.5 * (np.float32(event["theta_min"]) + np.float32(event["theta_max"]))
        phi = 0.5 * (np.float32(event["phi_min"]) + np.float32(event["phi_max"]))
        rotation = np.degrees(event["angle_array_x_magnetic_north"])
        showers = int(run["n_showers"])
        return {
            "primary_particle": PrimaryParticle("corsika7_id", int(event["particle_id"])),
            "zenith_angle": round(float(zenith), 2),
            "azimuth_angle": round(float((geographic_to_corsika_azimuth(phi) + rotation) % 360), 2),
            "zenith_min": float(np.float32(event["theta_min"])),
            "viewcone_max": float(np.float32(event["viewcone_outer_angle"])),
            "shower_events": showers,
            "mc_events": showers * int(event["n_reuse"]),
            "observation_levels": [
                height / 100 for height in run["observation_height"][: run["n_observation_levels"]]
            ],
        }

    def iter_records(self, file_id=0):
        """Yield (table name, row) pairs in reduced-table units.

        Shower rows are retained until all reuse positions have been read.
        Memory use is bounded by one shower's reuse count.

        Parameters
        ----------
        file_id : int
            Identifier assigned by the reduced-table writer.

        Yields
        ------
        tuple
            Table name and dictionary matching the reduced-table schema.
        """
        run_info = {}
        with EventIOFile(self.file_name) as eventio_file:
            for obj in eventio_file:
                if isinstance(obj, MCShower | iact.EventHeader):
                    yield from (("SHOWERS", row) for row in self.shower_data)
                    self.shower_data = []
                self._process_eventio_object(obj, file_id, self.file_name, run_info)
                yield from (("TRIGGERS", row) for row in self.trigger_data)
                self.trigger_data = []
        yield from (("SHOWERS", row) for row in self.shower_data)
        self.shower_data = []
        self._process_file_info(file_id, self.file_name, run_info or None)
        self._record_input_metadata(file_id, self.file_name, run_info or None)
        yield "FILE_INFO", self.file_info[0]

    def read_metadata(self):
        """Return metadata collected while iterating reduced records."""
        return self.input_metadata[0]

    def _process_eventio_object(self, eventio_object, file_id, file, run_info):
        """Process one EventIO object and update accumulated records."""
        if isinstance(eventio_object, RunHeader):
            self._current_run_header = eventio_object.parse()
            run_info.update(self._current_run_header)
            self._current_simtel_metadata = read_sim_telarray_metadata(file)
            self.telescope_id_to_name = get_sim_telarray_telescope_id_to_telescope_name_mapping(
                file
            )
        elif isinstance(eventio_object, MCRunHeader):
            self._current_mc_run_header = self._process_mc_run_header(eventio_object)
            run_info.update(self._current_mc_run_header)
            if not self.telescope_id_to_name:
                self._current_simtel_metadata = read_sim_telarray_metadata(file)
                self.telescope_id_to_name = get_sim_telarray_telescope_id_to_telescope_name_mapping(
                    file
                )
        elif isinstance(eventio_object, MCShower):
            shower = self._process_mc_shower(eventio_object, file_id)
            if shower.get("primary_id") is not None:
                run_info.setdefault("primary_id", shower["primary_id"])
        elif isinstance(eventio_object, MCEvent):
            self._process_mc_event(eventio_object)
        elif isinstance(eventio_object, ArrayEvent):
            self._process_array_event(eventio_object, file_id)
        elif isinstance(eventio_object, iact.EventHeader):
            self._process_mc_shower_from_iact(eventio_object, file_id)

    def _process_mc_run_header(self, eventio_object):
        """Process MC run header (sim_telarray file)."""
        mc_head = eventio_object.parse()
        self.n_use = mc_head["n_use"]  # reuse factor n_use needed to extend the values below
        self._logger.info(f"Shower reuse factor: {self.n_use} (viewcone: {mc_head['viewcone']})")
        return mc_head

    def _record_input_metadata(self, file_id, file, run_info):
        """Retain rich provenance for one processed input file."""
        is_iact_file = not run_info
        if is_iact_file:
            self._current_run_header, _ = self.read_run_headers()
            run_info = self._current_run_header or {}
        global_metadata, telescope_metadata = self._current_simtel_metadata
        telescope_metadata = {
            str(telescope_id): {
                "telescope_name": self.telescope_id_to_name.get(telescope_id),
                "values": metadata,
            }
            for telescope_id, metadata in telescope_metadata.items()
        }
        self.input_metadata.append(
            {
                "file_id": file_id,
                "file_name": str(file),
                "run_number": run_info.get("run", run_info.get("run_number")),
                "run_header": self._current_run_header,
                "mc_run_header": self._current_mc_run_header,
                "sim_telarray_global_metadata": global_metadata,
                "sim_telarray_telescope_metadata": telescope_metadata,
            }
        )

        if is_iact_file:
            self.input_metadata[-1]["simulation_software"] = "corsika"

    def _process_file_info(self, file_id, file, run_info=None):
        """Process file information and append to file info list."""
        run_number = (
            run_info.get("run", run_info.get("run_number"))
            if run_info
            else self._get_corsika_run_number(file)
        )
        run_number = 0 if run_number is None else run_number
        if run_info:  # sim_telarray file
            corsika7_id = PrimaryParticle(
                particle_id_type="eventio_id",
                particle_id=run_info.get("primary_id", 1),
            ).corsika7_id
            nsb = self.get_nsb_level_from_sim_telarray_metadata(file)

            e_min, e_max = run_info["E_range"]
            view_cone_min, view_cone_max = run_info["viewcone"]
            core_min, core_max = run_info["core_range"]
            azimuth, el = np.degrees(run_info["direction"])
            zenith = 90.0 - el
            spectral_index = self._extract_spectral_index(run_info=run_info)
        else:  # CORSIKA IACT file
            run_header, event_header = get_corsika_run_and_event_headers(file)
            corsika7_id = int(event_header["particle_id"])
            e_min = event_header["energy_min"] / 1.0e3
            e_max = event_header["energy_max"] / 1.0e3
            zenith = np.degrees(event_header["zenith"])
            # Rotate to geographic north
            azimuth = _iact_geographic_azimuth(event_header)
            view_cone_min = event_header["viewcone_inner_angle"]
            view_cone_max = event_header["viewcone_outer_angle"]
            core_min = 0.0
            core_max = run_header["x_scatter"] / 1.0e2  # cm to m
            nsb = 0.0
            spectral_index = self._extract_spectral_index(event_header=event_header)

        self.file_info.append(
            {
                "file_name": str(file),
                "file_id": file_id,
                "run_number": run_number,
                "particle_id": corsika7_id,
                "spectral_index": spectral_index,
                "energy_min": e_min,
                "energy_max": e_max,
                "viewcone_min": view_cone_min,
                "viewcone_max": view_cone_max,
                "core_scatter_min": core_min,
                "core_scatter_max": core_max,
                "zenith": zenith,
                "azimuth": azimuth,
                "nsb_level": nsb,
            }
        )

    @staticmethod
    def _get_corsika_run_number(file):
        """Return the run number from a CORSIKA header when available."""
        try:
            run_header, event_header = get_corsika_run_and_event_headers(file)
            return event_header.get("run_number", run_header.get("run_number"))
        except KeyError, TypeError:
            return None

    @staticmethod
    def _extract_spectral_index(run_info=None, event_header=None):
        """Extract the original simulated spectral index from available headers."""
        for source in (run_info, event_header):
            if source is None:
                continue
            for key in ("energy_spectrum_slope", "spectral_index", "eslope"):
                value = source.get(key)
                if value is not None:
                    return float(value)
        return np.nan

    def _process_mc_shower(self, eventio_object, file_id):
        """
        Process MC shower from sim_telarray file and update shower event list.

        Duplicated entries 'self.n_use' times to match the number simulated events with
        different core positions.
        """
        shower = eventio_object.parse()

        self.shower_data.extend(
            {
                "shower_id": shower["shower"],
                "event_id": None,  # filled in _process_mc_event
                "file_id": file_id,
                "simulated_energy": shower["energy"],
                "x_core": None,  # filled in _process_mc_event
                "y_core": None,  # filled in _process_mc_event
                "shower_azimuth": np.degrees(shower["azimuth"]),
                "shower_altitude": np.degrees(shower["altitude"]),
                "area_weight": None,  # filled in _process_mc_event
            }
            for _ in range(self.n_use)
        )
        return shower

    def _process_mc_shower_from_iact(self, eventio_object, file_id):
        """
        Process MC shower from IACT file and update shower event list.

        Duplicated entries 'self.n_use' times to match the number simulated events with
        different core positions.
        """
        shower_header = eventio_object.parse()
        self.n_use = int(shower_header["n_reuse"])

        self.shower_data.extend(
            {
                "shower_id": shower_header["event_number"],
                "event_id": shower_header["event_number"] * 100 + i,
                "file_id": file_id,
                "simulated_energy": shower_header["total_energy"] / 1.0e3,
                "x_core": shower_header["reuse_x"][i] / 1.0e2,
                "y_core": shower_header["reuse_y"][i] / 1.0e2,
                "shower_azimuth": _iact_geographic_azimuth(shower_header),
                "shower_altitude": 90.0 - np.degrees(shower_header["zenith"]),
                "area_weight": 1.0,
            }
            for i in range(self.n_use)
        )

    def _process_mc_event(self, eventio_object):
        """
        Process MC event and update shower event list.

        Expected to be called n_use times after _process_shower.
        """
        event = eventio_object.parse()

        shower_data_index = len(self.shower_data) - self.n_use + event["event_id"] % 100

        try:
            if self.shower_data[shower_data_index]["shower_id"] != event["shower_num"]:
                raise IndexError
        except IndexError as exc:
            raise IndexError(
                f"Inconsistent shower and MC event data for shower id {event['shower_num']}"
            ) from exc

        self.shower_data[shower_data_index].update(
            {
                "event_id": event["event_id"],
                "x_core": event["xcore"],
                "y_core": event["ycore"],
                "area_weight": event["aweight"],
            }
        )

    def _process_array_event(self, eventio_object, file_id):
        """Process array event and update triggered event list."""
        altitudes = []
        azimuths = []
        telescopes = []

        for obj in eventio_object:
            if isinstance(obj, TriggerInformation):
                trigger_info = obj.parse()
                telescopes = (
                    trigger_info["triggered_telescopes"]
                    if len(trigger_info["triggered_telescopes"]) > 0
                    else []
                )
            if isinstance(obj, TrackingPosition):
                tracking_position = obj.parse()
                altitudes.append(np.degrees(tracking_position["altitude_raw"]))
                azimuths.append(np.degrees(tracking_position["azimuth_raw"]))

        if len(telescopes) > 0 and altitudes:
            self._fill_array_event(
                self._map_telescope_names(telescopes),
                altitudes,
                azimuths,
                eventio_object.event_id,
                file_id,
            )

    def _fill_array_event(self, telescopes, altitudes, azimuths, event_id, file_id):
        """Add array event triggered events with tracking positions."""
        self.trigger_data.append(
            {
                "shower_id": self.shower_data[-1]["shower_id"],
                "event_id": event_id,
                "file_id": file_id,
                "array_altitude": float(np.mean(altitudes)),
                "array_azimuth": float(np.degrees(calculate_circular_mean(np.deg2rad(azimuths)))),
                "telescope_list": ",".join(map(str, telescopes)),
                "telescope_list_common_id": ",".join(
                    [
                        str(get_common_identifier_from_array_element_name(tel, 0))
                        for tel in telescopes
                    ]
                ),
            }
        )

    def _map_telescope_names(self, telescope_ids):
        """
        Map sim_telarray telescopes IDs to CTAO array element names.

        Parameters
        ----------
        telescope_ids : list
            List of telescope IDs.

        Returns
        -------
        list
            List of telescope names corresponding to the IDs.
        """
        return [
            self.telescope_id_to_name.get(tel_id, f"Unknown_{tel_id}") for tel_id in telescope_ids
        ]

    def get_nsb_level_from_sim_telarray_metadata(self, file):
        """
        Return NSB level from sim_telarray metadata.

        Falls back to preliminary NSB level if not found.

        Parameters
        ----------
        file : Path
            Path to the sim_telarray file.

        Returns
        -------
        float
            NSB level.
        """
        metadata, _ = read_sim_telarray_metadata(file)
        nsb_integrated_flux = metadata.get("nsb_integrated_flux")
        if nsb_integrated_flux is not None:
            try:
                return float(nsb_integrated_flux)
            except TypeError, ValueError:
                self._logger.warning(
                    f"Invalid nsb_integrated_flux value '{nsb_integrated_flux}' for {file}"
                )

        return self._get_nsb_level_from_file_name(str(file))

    def _get_nsb_level_from_file_name(self, file):
        """
        Return NSB level from file name.

        Hardwired values are used for "dark", "half", "full", and "moon" NSB levels.
        "moon" is treated as equivalent to "half" (0.835).
        Allows to read legacy sim_telarray files without 'nsb_integrated_flux'
        metadata field.

        Parameters
        ----------
        file : str
            File name to extract NSB level from.

        Returns
        -------
        float
            NSB level extracted from file name.

        Raises
        ------
        ValueError
            If no NSB level keyword is found in the file name.
        """
        nsb_levels = {"dark": 0.24, "half": 0.835, "full": 1.2}
        nsb_levels["moon"] = nsb_levels["half"]  # moon uses same level as half

        for key, value in nsb_levels.items():
            try:
                if key in file.lower():
                    self._logger.warning(f"NSB level set to hardwired value of {value} for {file}")
                    return value
            except AttributeError as exc:
                raise AttributeError("Invalid file name.") from exc

        raise ValueError(
            f"Cannot determine NSB level for '{file}': not found in metadata and "
            f"no recognised keyword ('dark', 'half', 'full', 'moon') in file name."
        )
