"""Generate lookup tables from EventIO simulation files and store them in an HDF5 file."""

import logging

import h5py
import numpy as np
from eventio import EventIOFile
from eventio.simtel import ArrayEvent, MCEvent, MCRunHeader, MCShower, TriggerInformation


class LookupTableGenerator:
    """
    A class to generate lookup tables from EventIO simulation files and store them in an HDF5 file.

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
        Initialize the LookupTableGenerator with input files, output file, and maximum file limit.

        Parameters
        ----------
        input_files : list
            List of input file paths to process.
        output_file : str
            Path to the output HDF5 file.
        max_files : int, optional
            Maximum number of files to process (default is 100).
        """
        self._logger = logging.getLogger(__name__)
        self.input_files = input_files
        self.output_file = output_file
        self.max_files = max_files

    def process_files(self):
        """Process the input files and store it in an HDF5 file."""
        with h5py.File(self.output_file, "w") as hdf:
            grp = hdf.create_group("data")
            dset_simulated = grp.create_dataset("simulated", (0,), maxshape=(None,), dtype="f4")
            dset_shower_id_triggered = grp.create_dataset(
                "shower_id_triggered", (0,), maxshape=(None,), dtype="i4"
            )
            dset_triggered = grp.create_dataset(
                "triggered_energies", (0,), maxshape=(None,), dtype="f4"
            )
            dset_num_triggered_telescopes = grp.create_dataset(
                "num_triggered_telescopes", (0,), maxshape=(None,), dtype="i4"
            )
            vlen_int_type = h5py.special_dtype(vlen=np.int16)
            dset_trigger_telescope_list_list = grp.create_dataset(
                "trigger_telescope_list_list",
                (0,),
                maxshape=(None,),
                dtype=vlen_int_type,
                chunks=True,
            )
            dset_core_x = grp.create_dataset("core_x", (0,), maxshape=(None,), dtype="f4")
            dset_core_y = grp.create_dataset("core_y", (0,), maxshape=(None,), dtype="f4")
            dset_file_names = grp.create_dataset(
                "file_names", (0,), maxshape=(None,), dtype=h5py.string_dtype(encoding="utf-8")
            )

            batch_size = 50000
            simulated, shower_id_triggered, triggered_energies = [], [], []
            num_triggered_telescopes, event_x_core, event_y_core = [], [], []
            trigger_telescope_list_list, file_names = [], []

            for i_file, file in enumerate(self.input_files[: self.max_files]):
                self._logger.info(f"Processing file {i_file+1}/{self.max_files}: {file}")
                file_names.append(file)

                with EventIOFile(file) as f:
                    for eventio_object in f:
                        if isinstance(eventio_object, MCRunHeader):
                            n_use = eventio_object.parse()["n_use"]

                        if isinstance(eventio_object, MCShower):
                            shower = eventio_object.parse()
                            simulated.extend(n_use * [shower["energy"]])

                        if isinstance(eventio_object, MCEvent):
                            event = eventio_object.parse()
                            event_x_core.append(event["xcore"])
                            event_y_core.append(event["ycore"])

                        if isinstance(eventio_object, ArrayEvent):
                            for i, obj in enumerate(eventio_object):
                                if i == 0 and isinstance(obj, TriggerInformation):
                                    trigger_info = obj.parse()
                                    telescopes = trigger_info["telescopes_with_data"]
                                    if len(telescopes) > 0:
                                        shower_id_triggered.append(shower["shower"])
                                        triggered_energies.append(shower["energy"])
                                        num_triggered_telescopes.append(len(telescopes))
                                        trigger_telescope_list_list.append(
                                            np.array(telescopes, dtype=np.int16)
                                        )

                if len(simulated) >= batch_size:
                    self._append_to_hdf5(dset_simulated, simulated)
                    self._append_to_hdf5(dset_shower_id_triggered, shower_id_triggered)
                    self._append_to_hdf5(dset_triggered, triggered_energies)
                    self._append_to_hdf5(dset_num_triggered_telescopes, num_triggered_telescopes)
                    self._append_to_hdf5(
                        dset_trigger_telescope_list_list, trigger_telescope_list_list
                    )
                    self._append_to_hdf5(dset_core_x, event_x_core)
                    self._append_to_hdf5(dset_core_y, event_y_core)
                    self._append_to_hdf5(dset_file_names, file_names)

                    simulated, shower_id_triggered, triggered_energies = [], [], []
                    num_triggered_telescopes, event_x_core, event_y_core = [], [], []
                    trigger_telescope_list_list, file_names = [], []

            self._append_to_hdf5(dset_simulated, simulated)
            self._append_to_hdf5(dset_shower_id_triggered, shower_id_triggered)
            self._append_to_hdf5(dset_triggered, triggered_energies)
            self._append_to_hdf5(dset_num_triggered_telescopes, num_triggered_telescopes)
            self._append_to_hdf5(dset_trigger_telescope_list_list, trigger_telescope_list_list)
            self._append_to_hdf5(dset_core_x, event_x_core)
            self._append_to_hdf5(dset_core_y, event_y_core)
            self._append_to_hdf5(dset_file_names, file_names)

    @staticmethod
    def _append_to_hdf5(dataset, data):
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
        except Exception as exc:
            raise ValueError("An error occurred while reading the HDF5 file") from exc
