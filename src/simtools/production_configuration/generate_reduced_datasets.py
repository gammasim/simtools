"""Generate a reduced dataset from given simulation event list and save the output to file."""

import logging

import h5py
import numpy as np
from eventio import EventIOFile
from eventio.simtel import ArrayEvent, MCEvent, MCRunHeader, MCShower, TriggerInformation


class ReducedDatasetGenerator:
    """
    Generate a reduced dataset from given simulation event list and save the output to file.

    Attributes
    ----------
    input_files : list
        List of input file paths to process.
    output_file : str
        Path to the output HDF5 file.
    max_files : int, optional
        Maximum number of files to process (default is 100).
    """

    def __init__(self, input_files, output_file, max_files=100):
        """
        Initialize the ReducedDatasetGenerator with input files, output file, and max file limit.

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
        self.max_files = max_files
        self.shower = None
        self.n_use = None

    def process_files(self):
        """Process the input files and store it in an HDF5 file."""
        with h5py.File(self.output_file, "w") as hdf:
            grp = hdf.create_group("data")
            datasets = self._create_datasets(grp)
            data_lists = self._initialize_data_lists()

            for i_file, file in enumerate(self.input_files[: self.max_files]):
                self._logger.info(f"Processing file {i_file + 1}/{self.max_files}: {file}")
                data_lists["file_names"].append(str(file))
                self._process_file(file, data_lists)

                if len(data_lists["simulated"]) >= 50000:
                    self._append_all_datasets(datasets, data_lists)
                    self._reset_data_lists(data_lists)

            self._append_all_datasets(datasets, data_lists)

    def _create_datasets(self, grp):
        """Create HDF5 datasets and add units as attributes."""
        vlen_int_type = h5py.special_dtype(vlen=np.int16)

        datasets = {
            "simulated": grp.create_dataset(
                "simulated", (0,), maxshape=(None,), dtype="f4", compression="gzip"
            ),
            "shower_id_triggered": grp.create_dataset(
                "shower_id_triggered", (0,), maxshape=(None,), dtype="i4", compression="gzip"
            ),
            "triggered_energies": grp.create_dataset(
                "triggered_energies", (0,), maxshape=(None,), dtype="f4", compression="gzip"
            ),
            "num_triggered_telescopes": grp.create_dataset(
                "num_triggered_telescopes", (0,), maxshape=(None,), dtype="i4", compression="gzip"
            ),
            "trigger_telescope_list_list": grp.create_dataset(
                "trigger_telescope_list_list",
                (0,),
                maxshape=(None,),
                dtype=vlen_int_type,
                chunks=True,
                compression="gzip",
            ),
            "core_x": grp.create_dataset(
                "core_x", (0,), maxshape=(None,), dtype="f4", compression="gzip"
            ),
            "core_y": grp.create_dataset(
                "core_y", (0,), maxshape=(None,), dtype="f4", compression="gzip"
            ),
            "file_names": grp.create_dataset(
                "file_names",
                (0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
                compression="gzip",
            ),
            "shower_sim_azimuth": grp.create_dataset(
                "shower_sim_azimuth", (0,), maxshape=(None,), dtype="f4", compression="gzip"
            ),
            "shower_sim_altitude": grp.create_dataset(
                "shower_sim_altitude", (0,), maxshape=(None,), dtype="f4", compression="gzip"
            ),
            "array_altitudes": grp.create_dataset(
                "array_altitudes", (0,), maxshape=(None,), dtype="f4", compression="gzip"
            ),
            "array_azimuths": grp.create_dataset(
                "array_azimuths", (0,), maxshape=(None,), dtype="f4", compression="gzip"
            ),
        }

        datasets["simulated"].attrs["units"] = "TeV"
        datasets["shower_id_triggered"].attrs["units"] = "ID"
        datasets["triggered_energies"].attrs["units"] = "TeV"
        datasets["num_triggered_telescopes"].attrs["units"] = "count"
        datasets["trigger_telescope_list_list"].attrs["units"] = "ID"
        datasets["core_x"].attrs["units"] = "m"
        datasets["core_y"].attrs["units"] = "m"
        datasets["shower_sim_azimuth"].attrs["units"] = "rad"
        datasets["shower_sim_altitude"].attrs["units"] = "rad"
        datasets["array_altitudes"].attrs["units"] = "m"
        datasets["array_azimuths"].attrs["units"] = "rad"

        return datasets

    def _initialize_data_lists(self):
        """Initialize data lists."""
        return {
            "simulated": [],
            "shower_id_triggered": [],
            "triggered_energies": [],
            "num_triggered_telescopes": [],
            "core_x": [],
            "core_y": [],
            "trigger_telescope_list_list": [],
            "file_names": [],
            "shower_sim_azimuth": [],
            "shower_sim_altitude": [],
            "array_altitudes": [],
            "array_azimuths": [],
        }

    def _process_file(self, file, data_lists):
        """Process a single file and update data lists."""
        with EventIOFile(file) as f:
            array_altitude = None
            array_azimuth = None
            for eventio_object in f:
                if isinstance(eventio_object, MCRunHeader):
                    self._process_mc_run_header(eventio_object, data_lists)
                elif isinstance(eventio_object, MCShower):
                    self._process_mc_shower(
                        eventio_object, data_lists, array_altitude, array_azimuth
                    )
                elif isinstance(eventio_object, MCEvent):
                    self._process_mc_event(eventio_object, data_lists)
                elif isinstance(eventio_object, ArrayEvent):
                    self._process_array_event(eventio_object, data_lists)

    def _process_mc_run_header(self, eventio_object, data_lists):
        """Process MC run header and update data lists."""
        mc_head = eventio_object.parse()
        self.n_use = mc_head["n_use"]  # reuse factor n_use needed to extend the values below
        array_altitude = np.mean(mc_head["alt_range"])
        array_azimuth = np.mean(mc_head["az_range"])
        data_lists["array_altitudes"].extend(self.n_use * [array_altitude])
        data_lists["array_azimuths"].extend(self.n_use * [array_azimuth])

    def _process_mc_shower(self, eventio_object, data_lists, array_altitude, array_azimuth):
        """Process MC shower and update data lists."""
        self.shower = eventio_object.parse()
        data_lists["simulated"].extend(self.n_use * [self.shower["energy"]])
        data_lists["shower_sim_azimuth"].extend(self.n_use * [self.shower["azimuth"]])
        data_lists["shower_sim_altitude"].extend(self.n_use * [self.shower["altitude"]])
        data_lists["array_altitudes"].extend(self.n_use * [array_altitude])
        data_lists["array_azimuths"].extend(self.n_use * [array_azimuth])

    def _process_mc_event(self, eventio_object, data_lists):
        """Process MC event and update data lists."""
        event = eventio_object.parse()
        data_lists["core_x"].append(event["xcore"])
        data_lists["core_y"].append(event["ycore"])

    def _process_array_event(self, eventio_object, data_lists):
        """Process array event and update data lists."""
        for i, obj in enumerate(eventio_object):
            if i == 0 and isinstance(obj, TriggerInformation):
                self._process_trigger_information(obj, data_lists)

    def _process_trigger_information(self, trigger_info, data_lists):
        """Process trigger information and update data lists."""
        trigger_info = trigger_info.parse()
        telescopes = trigger_info["telescopes_with_data"]
        if len(telescopes) > 0:
            data_lists["shower_id_triggered"].append(self.shower["shower"])
            data_lists["triggered_energies"].append(self.shower["energy"])
            data_lists["num_triggered_telescopes"].append(len(telescopes))
            data_lists["trigger_telescope_list_list"].append(np.array(telescopes, dtype=np.int16))

    def _append_to_hdf5(self, dataset, data):
        """
        Append data to an HDF5 dataset, resizing it as needed.

        Parameters
        ----------
        dataset : h5py.Dataset
            The dataset to append to.
        data : list
            The data to append.
        """
        if len(data) > 0:
            dataset.resize((dataset.shape[0] + len(data),))
            dataset[-len(data) :] = data

    def _append_all_datasets(self, datasets, data_lists):
        """
        Append all data to the respective HDF5 datasets.

        Parameters
        ----------
        datasets : dict
            Dictionary containing HDF5 datasets.
        data_lists : dict
            Dictionary containing lists of data to append.
        """
        for key, dataset in datasets.items():
            self._append_to_hdf5(dataset, data_lists[key])

    def _reset_data_lists(self, data_lists):
        """Reset data lists during batch processing."""
        for key in data_lists:
            data_lists[key] = []

    def print_hdf5_file(self):
        """Print information about the datasets in the generated HDF5 file."""
        try:
            with h5py.File(self.output_file, "r") as hdf:
                print("Datasets in file:")
                for key in hdf["data"].keys():
                    dset = hdf["data"][key]
                    print(f"- {key}: shape={dset.shape}, dtype={dset.dtype}")

                    # Print first 5 values each
                    print(f"  First 5 values: {dset[:5]}")

                    # Print units if available
                    units = dset.attrs.get("units", "N/A")
                    print(f"  Units: {units}")
        except Exception as exc:
            raise ValueError("An error occurred while reading the HDF5 file") from exc
