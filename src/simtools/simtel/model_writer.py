"""Write sim_telarray configuration and native tables from simulation models."""

import logging
from copy import deepcopy
from pathlib import Path

import astropy.units as u

from simtools import settings
from simtools.data_model import schema
from simtools.data_model.table_asset import get_simtel_serialization
from simtools.model_repository.asset_names import get_simtel_table_file_name
from simtools.simtel import simtel_seeds, table_serializers
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.utils import names


class SimtelModelWriter:
    """Export a resolved model in sim_telarray configuration formats.

    Parameters
    ----------
    model : ModelParameter or ArrayModel
        Model providing validated parameters, tables, and output paths.
    """

    def __init__(self, model):
        self.model = model
        self._logger = logging.getLogger(__name__)
        self._serialized_tables = {}
        self.config_writer = None
        self.single_mirror_list_file_paths = {}
        self._telescope_model_files_exported = False
        self._array_model_file_exported = False

    def write_sim_telarray_config_file(self, additional_models=None, label=None):
        """
        Write the sim_telarray configuration file.

        Parameters
        ----------
        additional_models: TelescopeModel or SiteModel
            Model object for additional parameter to be written to the config file.
        label: str or None
            Optional label override used for output file naming.
        """
        parameters = self.model.parameters.copy()
        parameters.update(self.model.get_simulation_software_parameters("sim_telarray") or {})
        self.model.export_model_files(update_if_necessary=True)
        if (
            "correct_nsb_spectrum_to_telescope_altitude"
            in self.model.get_simulation_software_parameters("sim_telarray")
        ):
            self.export_nsb_spectrum_to_telescope_altitude_correction_file(
                model_directory=self.model.config_file_directory
            )
        self._add_additional_models(additional_models, parameters)

        # Ensure the writer label matches the config file naming label
        self._load_simtel_config_writer(label=label if label is not None else self.model.label)
        self.config_writer.write_telescope_config_file(
            config_file_path=self.model.config_file_path,
            parameters=parameters,
        )
        return self.model.config_file_path

    def _add_additional_models(self, additional_models, parameters):
        """Add additional models to the current model parameters."""
        if additional_models is None:
            return

        if isinstance(additional_models, dict):
            for additional_model in additional_models.values():
                self._add_additional_models(additional_model, parameters)
            return

        parameters.update(additional_models.parameters)
        parameters.update(
            {
                name: parameter
                for name, parameter in (
                    additional_models.get_simulation_software_parameters("sim_telarray") or {}
                ).items()
                if names.is_global_sim_telarray_parameter(name)
            }
        )
        additional_models.export_model_files(
            self.model.config_file_directory, update_if_necessary=True
        )

    def _load_simtel_config_writer(self, label=None):
        """Load the SimtelConfigWriter object."""
        desired_label = self.model.label if label is None else label
        if self.config_writer is None or desired_label != self.config_writer.label:
            self.config_writer = SimtelConfigWriter(
                site=self.model.site,
                telescope_model_name=self.model.name,
                telescope_design_model=self.model.design_model,
                model_version=self.model.model_version,
                label=desired_label,
                model_reader=self.model.model_reader,
            )

    def export_nsb_spectrum_to_telescope_altitude_correction_file(self, model_directory):
        """
        Export the NSB correction table and its native source file.

        Camera-efficiency calculations correct the NSB spectrum from the original altitude used in
        the Benn & Ellison model to the telescope altitude. The correction table is a
        simulation-software parameter and is not included in the ordinary model-parameter export.
        This method exports both the source file and the native table used by the calculation.

        Parameters
        ----------
        model_directory: Path
            Model directory to export the file to.
        """
        parameter_name = "correct_nsb_spectrum_to_telescope_altitude"
        correction_parameters = self.model.get_simulation_software_parameters("sim_telarray")
        if parameter_name not in correction_parameters:
            return
        parameter = deepcopy(correction_parameters[parameter_name])
        parameter["parameter"] = parameter_name
        parameter.setdefault("parameter_version", Path(parameter["value"]).stem.rsplit("-", 1)[-1])
        parameter.setdefault("instrument", self.model.design_model or self.model.name)
        parameter.setdefault("site", self.model.site)
        parameter["file"] = True
        self.model.model_reader.export_model_files(
            parameters={parameter_name: parameter},
            dest=model_directory,
        )

        self._export_ecsv_as_simtel_table(
            parameter_name,
            parameter,
            model_directory,
            table_format="atmospheric_transmission",
        )

    def _export_ecsv_as_simtel_table(
        self, parameter_name, parameter, model_directory, table_format, output_name=None
    ):
        """Export an ECSV model asset in the native sim_telarray table format."""
        if Path(parameter["value"]).suffix.lower() != ".ecsv":
            return None

        schema_data = schema.get_model_parameter_schema(
            parameter_name, parameter.get("model_parameter_schema_version")
        )
        contract = get_simtel_serialization(schema_data)
        contract["table_format"] = table_format
        generated_output_name = get_simtel_table_file_name(parameter)
        shared_output = generated_output_name is not None and (
            output_name is None or output_name == generated_output_name
        )
        output_name = output_name or generated_output_name
        output_name = output_name or f"{parameter_name}-{self.model.name}.dat"
        output_path = Path(model_directory) / output_name
        cache_key = repr((parameter_name, parameter, table_format, output_name))
        if output_path.is_file() and (
            shared_output or self._serialized_tables.get(output_path) == cache_key
        ):
            return output_path.name
        table = self.model.model_reader.get_parameter_table(parameter)
        result = table_serializers.write_simtel_table(
            table,
            model_directory,
            contract=contract,
            output_name=output_name,
        )
        self._serialized_tables[output_path] = cache_key
        return result

    def export_model_parameter_as_simtel_file(
        self, parameter_name, model_directory, table_format, output_name
    ):
        """Export an ECSV model parameter in the native sim_telarray format."""
        parameter = self.model.parameters[parameter_name].copy()
        return self._export_ecsv_as_simtel_table(
            parameter_name,
            parameter,
            model_directory,
            table_format=table_format,
            output_name=output_name,
        )

    def export_simtel_telescope_config_files(self):
        """Export sim_telarray configuration files for all telescopes into the model directory."""
        exported_models = []
        for tel_model in self.model.telescope_models.values():
            name = tel_model.name
            if name not in exported_models:
                tel_model.write_sim_telarray_config_file(
                    additional_models=self.model.calibration_models.get(tel_model.name)
                )
                exported_models.append(name)
            else:
                self._logger.debug(
                    f"Configuration file for telescope {name} already exists - skipping"
                )

        self._telescope_model_files_exported = True

    def export_sim_telarray_config_file(self):
        """Export sim_telarray configuration file for the array into the model directory."""
        self.model.site_model.export_model_files()

        self._logger.info(f"Writing array configuration file into {self.model.config_file_path}")
        simtel_writer = SimtelConfigWriter(
            site=self.model.site_model.site,
            layout_name=self.model.layout_name,
            model_version=self.model.model_version,
            label=self.model.label,
            model_reader=self.model.model_reader,
        )
        simtel_writer.write_array_config_file(
            config_file_path=self.model.config_file_path,
            telescope_model=self.model.telescope_models,
            site_model=self.model.site_model,
            additional_metadata=self._get_additional_simtel_metadata(),
        )
        self._array_model_file_exported = True

    def export_all_simtel_config_files(self):
        """
        Export sim_telarray config file for the array and for each individual telescope.

        Config files are exported into the output model directory.
        """
        if not self._telescope_model_files_exported:
            self.export_simtel_telescope_config_files()
        if not self._array_model_file_exported:
            self.export_sim_telarray_config_file()

    def _get_additional_simtel_metadata(self):
        """
        Collect additional metadata to be included in sim_telarray output.

        Returns
        -------
        dict
            Dictionary with additional metadata.
        """
        metadata = {
            "nsb_integrated_flux": self.model.site_model.get_nsb_integrated_flux(),
        }
        for metadata_key, args_key in {
            "primary": "primary",
            "azimuth_angle": "azimuth_angle",
            "zenith_angle": "zenith_angle",
            "ha_angle": "ha",
            "dec_angle": "dec",
        }.items():
            value = settings.config.args.get(args_key)
            if value is not None:
                metadata[metadata_key] = (
                    value.to_value(u.deg) if hasattr(value, "to_value") else value
                )
        return metadata

    def write_config_file(self, additional_models=None, label=None):
        """Write telescope or calibration configuration in sim_telarray format.

        Parameters
        ----------
        additional_models : ModelParameter or dict, optional
            Additional models included in the configuration.
        label : str, optional
            File naming label.

        Returns
        -------
        pathlib.Path
            Written native configuration file.
        """
        return self.write_sim_telarray_config_file(additional_models, label)

    def export_config_files(self):
        """Write array and telescope configurations in sim_telarray format."""
        return self.export_all_simtel_config_files()

    def export_atmospheric_transmission_file(self, model_directory):
        """
        Export the atmospheric profile source and sim_telarray table files.

        Parameters
        ----------
        model_directory: Path
            Model directory to export the file to.
        """
        atmospheric_profile = self.model.parameters["atmospheric_profile"].copy()
        atmospheric_profile["qualify_filename"] = False
        self.model.model_reader.export_model_files(
            parameters={"atmospheric_profile": atmospheric_profile},
            dest=model_directory,
        )
        self._export_ecsv_as_simtel_table(
            parameter_name="atmospheric_profile",
            parameter=atmospheric_profile,
            model_directory=model_directory,
            table_format="plain",
            output_name=self.get_atmospheric_profile_file_name(atmospheric_profile),
        )

    def get_atmospheric_profile_file_name(self, parameter=None):
        """Return the native filename used for the CORSIKA atmospheric profile."""
        parameter = (parameter or self.model.parameters["atmospheric_profile"]).copy()
        parameter["qualify_filename"] = False
        file_name = get_simtel_table_file_name(parameter)
        if file_name is not None:
            return file_name
        value_path = Path(parameter["value"])
        if value_path.suffix.lower() == ".ecsv":
            return value_path.with_suffix(".dat").name
        return value_path.name

    def export_single_mirror_list_file(self, mirror_number: int, set_focal_length_to_zero: bool):
        """
        Export a mirror list file with a single mirror in it.

        Parameters
        ----------
        mirror_number: int
            Number index of the mirror.
        set_focal_length_to_zero: bool
            Set the focal length to zero if True.
        """
        if mirror_number > self.model.mirrors.number_of_mirrors:
            self._logger.error("mirror_number > number_of_mirrors")
            return

        file_name = names.simtel_single_mirror_list_file_name(
            self.model.site,
            self.model.name,
            self.model.model_version,
            mirror_number,
            self.model.label,
        )
        self.single_mirror_list_file_paths[mirror_number] = (
            self.model.config_file_directory.joinpath(file_name)
        )

        # Using SimtelConfigWriter
        self._load_simtel_config_writer()
        self.config_writer.write_single_mirror_list_file(
            mirror_number,
            self.model.mirrors,
            self.single_mirror_list_file_paths[mirror_number],
            set_focal_length_to_zero,
        )

    def initialize_seeds(self, zenith_angle=None, azimuth_angle=None):
        """Initialize sim_telarray seeds for instrument and shower simulations."""
        self.model.sim_telarray_seed = simtel_seeds.SimtelSeeds(
            output_path=self.model.get_config_directory(),
            site=self.model.site_model.site,
            model_version=self.model.model_version,
            zenith_angle=zenith_angle,
            azimuth_angle=azimuth_angle,
        )

    def get_single_mirror_list_file(self, mirror_number, set_focal_length_to_zero=False):
        """Return the native single-mirror file, writing it from the mirror model.

        Parameters
        ----------
        mirror_number : int
            Mirror index.
        set_focal_length_to_zero : bool
            Write zero focal length for the selected mirror.

        Returns
        -------
        pathlib.Path
            Single-mirror configuration file.
        """
        self.export_single_mirror_list_file(mirror_number, set_focal_length_to_zero)
        return self.single_mirror_list_file_paths[mirror_number]
