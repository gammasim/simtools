"""Generate a reduced dataset from given simulation event list and save the output to file."""

import logging

import numpy as np
import tables
from ctapipe.core import Container, Field
from ctapipe.io import HDF5TableWriter
from eventio import EventIOFile
from eventio.simtel import (
    ArrayEvent,
    MCEvent,
    MCRunHeader,
    MCShower,
    TrackingPosition,
    TriggerInformation,
)

DEFAULT_FILTERS = tables.Filters(complevel=5, complib="zlib", shuffle=True, bitshuffle=False)


class ReducedDatasetContainer(Container):
    """Container for reduced dataset information."""

    shower_id = Field(None, "Shower ID")
    sim_energy = Field(None, "Simulated energy")
    core_x = Field(None, "X-coordinate of the shower core")
    core_y = Field(None, "Y-coordinate of the shower core")
    area_weight = Field(None, "Area weighting factor")
    shower_sim_azimuth = Field(None, "Simulated azimuth angle of the shower")
    shower_sim_altitude = Field(None, "Simulated altitude angle of the shower")


class TriggeredShowerContainer(Container):
    """Container for triggered shower information."""

    triggered_id = Field(None, "Event ID for triggered event")
    triggered_energy = Field(None, "Shower energy for triggered event")
    array_altitudes = Field(None, "Altitudes for the array")
    array_azimuths = Field(None, "Azimuths for the array")


class FileNamesContainer(Container):
    """Container for file names."""

    file_names = Field(None, "Input file names")


class MCEventExtractor:
    """
    Generate a reduced dataset from given simulation event list and save the output to file.

    Attributes
    ----------
    input_files : list
        List of input file paths to process.
    output_file : str
        Path to the output HDF5 file.
    max_files : int, optional
        Maximum number of files to process.
    """

    def __init__(self, input_files, output_file, max_files=100):
        """
        Initialize the MCEventExtractor with input files, output file, and max file limit.

        Parameters
        ----------
        input_files : list
            List of input file paths to process.
        output_file : str
            Path to the output HDF5 file.
        max_files : int, optional
            Maximum number of files to process.
        """
        self._logger = logging.getLogger(__name__)
        self.input_files = input_files
        self.output_file = output_file
        self.max_files = max_files if max_files < len(input_files) else len(input_files)
        self.shower = None
        self.n_use = None
        self.shower_id_offset = 0

    def process_files(self):
        """Process the input files and store them in an HDF5 file."""
        if not self.input_files:
            self._logger.warning("No input files provided.")
            return

        data_lists = self._initialize_data_lists()
        self.shower_id_offset = 0
        # Process the first file in write mode
        self._logger.info(f"Processing file 1/{self.max_files}: {self.input_files[0]}")
        self._process_file(self.input_files[0], data_lists, str(self.input_files[0]))
        self._write_all_data(data_lists, mode="w")
        self.shower_id_offset = len(data_lists["sim_energy"])
        self._reset_data_lists(data_lists)

        # Process remaining files in append mode
        for i_file, file in enumerate(self.input_files[1 : self.max_files], start=2):
            self._logger.info(f"Processing file {i_file}/{self.max_files}: {file}")
            self._process_file(file, data_lists, str(file))
            if len(data_lists["sim_energy"]) >= 1e7:
                self._write_all_data(data_lists, mode="a")
                self.shower_id_offset += len(data_lists["sim_energy"])
                self._reset_data_lists(data_lists)

        # Final write for any remaining data
        self._write_all_data(data_lists, mode="a")

    def _write_all_data(self, data_lists, mode):
        """Write all data sections at once helper method."""
        self._write_data(data_lists, mode=mode)
        self._write_variable_length_data(data_lists["trigger_telescope_list_list"], mode="a")
        self._write_file_names(data_lists["file_names"], mode="a")

    def _write_file_names(self, file_names, mode="a"):
        """Write file names to HDF5 file."""
        print("file_names", file_names)
        with HDF5TableWriter(
            self.output_file, group_name="data", mode=mode, filters=DEFAULT_FILTERS
        ) as writer:
            file_names_container = FileNamesContainer()
            for file_name in file_names:
                file_names_container.file_names = file_name
                writer.write(table_name="file_names", containers=[file_names_container])

    def _write_variable_length_data(self, trigger_telescope_list_list, mode="a"):
        """Write variable-length array data to HDF5 file."""
        with tables.open_file(self.output_file, mode=mode) as f:
            if "trigger_telescope_list_list" in f.root.data:
                vlarray = f.root.data.trigger_telescope_list_list
            else:
                vlarray = f.create_vlarray(
                    f.root.data,
                    "trigger_telescope_list_list",
                    tables.Int16Atom(),
                    "List of triggered telescope IDs",
                )

            for item in trigger_telescope_list_list:
                vlarray.append(item)

    def _initialize_data_lists(self):
        """Initialize data lists."""
        return {
            "shower_id": [],
            "sim_energy": [],
            "triggered_id": [],
            "triggered_energy": [],
            "core_x": [],
            "core_y": [],
            "area_weight": [],
            "shower_sim_azimuth": [],
            "shower_sim_altitude": [],
            "trigger_telescope_list_list": [],
            "file_names": [],
            "array_altitudes": [],
            "array_azimuths": [],
        }

    def _process_file(self, file, data_lists, file_name):
        """Process a single file and update data lists."""
        with EventIOFile(file) as f:
            for eventio_object in f:
                if isinstance(eventio_object, MCRunHeader):
                    self._process_mc_run_header(eventio_object)
                elif isinstance(eventio_object, MCShower):
                    self._process_mc_shower(eventio_object, data_lists)
                elif isinstance(eventio_object, MCEvent):
                    self._process_mc_event(eventio_object, data_lists)
                elif isinstance(eventio_object, ArrayEvent):
                    self._process_array_event(eventio_object, data_lists)
            data_lists["file_names"].extend([file_name])

    def _process_mc_run_header(self, eventio_object):
        """Process MC run header and update data lists."""
        mc_head = eventio_object.parse()
        self.n_use = mc_head["n_use"]  # reuse factor n_use needed to extend the values below
        self._logger.info(f"Shower reuse factor: {self.n_use}")

    def _process_mc_shower(self, eventio_object, data_lists):
        """
        Process MC shower and update data lists.

        Duplicated entries 'self.n_use' times to match the number simulated events with
        different core positions.
        """
        self.shower = eventio_object.parse()
        data_lists["sim_energy"].extend(self.n_use * [self.shower["energy"]])
        data_lists["shower_sim_azimuth"].extend(self.n_use * [self.shower["azimuth"]])
        data_lists["shower_sim_altitude"].extend(self.n_use * [self.shower["altitude"]])

    def _process_mc_event(self, eventio_object, data_lists):
        """Process MC event and update data lists."""
        event = eventio_object.parse()
        data_lists["shower_id"].append(event["shower_num"])
        data_lists["core_x"].append(event["xcore"])
        data_lists["core_y"].append(event["ycore"])
        data_lists["area_weight"].append(event["aweight"])

    def _process_array_event(self, eventio_object, data_lists):
        """Process array event and update data lists."""
        tracking_positions = []
        previous_index = -1

        for i, obj in enumerate(eventio_object):
            if isinstance(obj, TriggerInformation):
                self._process_trigger_information(obj, data_lists)

            if isinstance(obj, TrackingPosition):
                tracking_position = obj.parse()
                tracking_positions.append(
                    {
                        "altitude": tracking_position["altitude_raw"],
                        "azimuth": tracking_position["azimuth_raw"],
                    }
                )

            if i < previous_index:
                self._process_tracking_positions(tracking_positions, data_lists)
                tracking_positions = []  # Reset for the next shower

            previous_index = i

        if tracking_positions:
            self._process_tracking_positions(tracking_positions, data_lists)

    def _process_tracking_positions(self, tracking_positions, data_lists):
        """
        Process the collected tracking positions and update data lists.

        For events with telescopes pointing in different directions, append
        mean pointing directions.

        Parameters
        ----------
        tracking_positions : list of dict
            List of tracking positions with "altitude" and "azimuth" keys.
        data_lists : dict
            Data lists to update.
        """
        altitudes = [pos["altitude"] for pos in tracking_positions]
        azimuths = [pos["azimuth"] for pos in tracking_positions]
        if isinstance(altitudes[0], list):
            altitudes = altitudes[0]
            azimuths = azimuths[0]
        # Check if all tracking positions are the same
        if np.allclose(altitudes, altitudes[0], atol=1e-5) and np.allclose(
            azimuths, azimuths[0], atol=1e-5
        ):
            data_lists["array_altitudes"].append(altitudes[0])
            data_lists["array_azimuths"].append(azimuths[0])
        else:  # append the mean telescope tracking positions for each triggered event
            self._logger.info("Telescopes have different tracking positions, applying mean.")
            data_lists["array_altitudes"].append(np.mean(altitudes))
            # TODO fix mean of azimuth values?
            data_lists["array_azimuths"].append(np.mean(azimuths))

    def _process_trigger_information(self, trigger_info, data_lists):
        """Process trigger information and update data lists."""
        trigger_info = trigger_info.parse()
        telescopes = trigger_info["telescopes_with_data"]
        if len(telescopes) > 0:
            # add offset to obtained unique shower IDs among all files
            data_lists["triggered_id"].append(self.shower["shower"] + self.shower_id_offset)
            data_lists["triggered_energy"].append(self.shower["energy"])
            data_lists["trigger_telescope_list_list"].append(np.array(telescopes, dtype=np.int16))

    def _write_data(self, data_lists, mode="a"):
        """Write data to HDF5 file using HDF5TableWriter."""
        with HDF5TableWriter(
            self.output_file, group_name="data", mode=mode, filters=DEFAULT_FILTERS
        ) as writer:
            # Write reduced dataset container
            reduced_container = ReducedDatasetContainer()

            for i in range(len(data_lists["sim_energy"])):
                reduced_container.shower_id = data_lists["shower_id"][i]
                reduced_container.sim_energy = data_lists["sim_energy"][i]
                reduced_container.core_x = data_lists["core_x"][i]
                reduced_container.core_y = data_lists["core_y"][i]
                reduced_container.area_weight = data_lists["area_weight"][i]
                reduced_container.shower_sim_azimuth = data_lists["shower_sim_azimuth"][i]
                reduced_container.shower_sim_altitude = data_lists["shower_sim_altitude"][i]

                writer.write(table_name="reduced_data", containers=[reduced_container])

            # Write triggered shower container
            triggered_container = TriggeredShowerContainer()

            for i in range(len(data_lists["triggered_id"])):
                triggered_container.triggered_id = data_lists["triggered_id"][i]
                triggered_container.triggered_energy = data_lists["triggered_energy"][i]
                triggered_container.array_altitudes = np.array(data_lists["array_altitudes"][i])
                triggered_container.array_azimuths = np.array(data_lists["array_azimuths"][i])
                writer.write(table_name="triggered_data", containers=[triggered_container])

    def _reset_data_lists(self, data_lists):
        """Reset data lists during batch processing."""
        data_lists.update({key: [] for key in data_lists})

    def print_dataset_information(self, n_events=5):
        """Print information about the datasets in the generated HDF5 file."""
        try:
            with tables.open_file(self.output_file, mode="r") as reader:
                print("Datasets in file:")
                for name, dataset in reader.root.data._v_children.items():  # pylint: disable=protected-access
                    print(f"- {name}: shape={dataset.shape}, dtype={dataset.dtype}")
                    print(f"  Length of dataset: {len(dataset)}")
                    print(f"  First {n_events} values: {dataset[:n_events]}")
        except Exception as exc:
            raise ValueError("An error occurred while reading the HDF5 file") from exc
