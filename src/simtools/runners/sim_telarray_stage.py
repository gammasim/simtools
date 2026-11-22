"""sim_telarray as a simulation stage (telescope and camera response)."""

import logging
import shlex

from simtools.job_execution.process_accounting import build_accounting_command
from simtools.model.mirrors import is_dual_mirror
from simtools.runners.stage_types import STDIN
from simtools.runners.stages import CatalogStage, efficiency_envelope_enabled, stages_with
from simtools.runtime import dependencies
from simtools.simtel.photon_bunch_file import bypass_optics_level
from simtools.simtel.simtel_runner import SIM_TELARRAY_ENV, sim_telarray_env_as_string
from simtools.simtel.simulator_array import SimulatorArray
from simtools.simulation_software import efficiency
from simtools.simulation_software.atmosphere import export_atmosphere_files, write_zero_transmission

#: Chain option selecting the photons sim_telarray simulates (values: see the catalog).
BYPASS_OPTICS_OPTION = "bypass_optics"
#: Option values: sim_telarray optics on the photons of the source (default); photons at the
#: camera entrance (Bypass_Optics 1 or 2 from the mirror class); camera only (Bypass_Optics 3).
BYPASS_NONE = "none"
BYPASS_CAMERA_ENTRANCE = "camera_entrance"
BYPASS_CAMERA_ONLY = "camera_only"
#: sim_telarray Bypass_Optics value of the camera-only simulation.
CAMERA_ONLY_BYPASS_LEVEL = 3
#: sim_telarray min_photons when an upstream stage thins photons. sim_telarray skips a telescope
#: event below min_photons, counted on the thinned photons, so the cut is disabled (the
#: min_photoelectrons cut still applies after the QE).
THINNED_MIN_PHOTONS = 0
#: Output directory of the zero-depth transmission table (sim_telarray, bypass 1 or 2). Separate
#: from the atmosphere stage: exporting the site table here must not overwrite its table.
ZERO_TRANSMISSION_DIRECTORY = "atmosphere_zero"
_logger = logging.getLogger(__name__)
_MODE_3_INCOMPATIBLE_TAGS = frozenset({"v2025-11-30-rc"})


def _bypass_level(mode, telescope_models):
    """
    Return sim_telarray's Bypass_Optics value for a bypass mode (None: sim_telarray optics).

    Camera-entrance photons take their level from the mirror class of the telescopes, which
    must be the same for all telescopes (Bypass_Optics is one setting of a sim_telarray run).

    Parameters
    ----------
    mode : str
        Value of the chain option ``bypass_optics``.
    telescope_models : dict
        Telescope models by telescope name.

    Raises
    ------
    ValueError
        If the telescopes do not share one mirror class for camera-entrance photons.
    """
    if mode == BYPASS_NONE:
        return None
    if mode == BYPASS_CAMERA_ONLY:
        return CAMERA_ONLY_BYPASS_LEVEL
    dual_mirror = {is_dual_mirror(telescope) for telescope in telescope_models.values()}
    if len(dual_mirror) != 1:
        raise ValueError(
            "Bypass_Optics needs the telescopes of a run to share one mirror class "
            f"(single or dual); telescopes: {sorted(telescope_models)}."
        )
    return bypass_optics_level(dual_mirror.pop())


def _zero_transmission(site_model, context):
    """Write the zero-depth transmission table of a site model; return its file."""
    directory = context.paths.get_output_directory(ZERO_TRANSMISSION_DIRECTORY)
    _profile, transmission = export_atmosphere_files(site_model, directory)
    return write_zero_transmission(transmission, directory)


def _transmission_file(zero_extinction, site_model, context):
    """Return the zero-depth transmission table if sim_telarray reads one (else None)."""
    return _zero_transmission(site_model, context) if zero_extinction else None


def _check_camera_only_build(context):
    """Reject known sim_telarray builds incompatible with camera-only simulation."""
    try:
        tag = dependencies.get_build_options(run_time=context.run_time, context=context).get(
            "simtel_tag"
        )
    except FileNotFoundError:
        tag = None
    if not tag:
        _logger.warning(
            "Cannot check camera-only sim_telarray compatibility: build tag unavailable"
        )
    elif tag in _MODE_3_INCOMPATIBLE_TAGS:
        raise ValueError(
            f"Bypass_Optics=3 is incompatible with sim_telarray {tag}; use a compatible build."
        )


def _array_instances(stage_class, inputs, label, mode, zero_extinction):
    """Return one sim_telarray stage per array model of a shower run."""
    envelope = efficiency_envelope_enabled(inputs)
    thinning = stages_with(inputs.chain.upstream_stages(stage_class.name), "thins_photons")
    if thinning:
        _logger.info(
            "Upstream %s thins photons: sim_telarray runs with min_photons=%s",
            ", ".join(thinning),
            THINNED_MIN_PHOTONS,
        )
    return [
        stage_class(
            simulation_parameters=parameters,
            array_model=array_model,
            label=label,
            context=inputs.context,
            bypass_optics=_bypass_level(mode, array_model.telescope_models),
            atmospheric_transmission=_transmission_file(
                zero_extinction, array_model.site_model, inputs.context
            ),
            efficiency_envelope=envelope,
            photons_thinned=bool(thinning),
        )
        for parameters, array_model in zip(inputs.simulation_parameters, inputs.array_models)
    ]


def _light_instances(stage_class, inputs, mode, zero_extinction):
    """Return the sim_telarray stage of a light-emission run (one telescope)."""
    calibration = inputs.calibration
    telescope_model = calibration.telescope_model
    return [
        LightSimTelArrayStage(
            stage_class,
            calibration,
            _bypass_level(mode, {telescope_model.name: telescope_model}),
            atmospheric_transmission=_transmission_file(
                zero_extinction, calibration.site_model, inputs.context
            ),
        )
    ]


class SimTelArrayStage(CatalogStage):
    """
    sim_telarray array simulation stage, one instance per array model.

    Parameters
    ----------
    simulation_parameters : SimulationParameters
        Physical run settings.
    array_model : ArrayModel
        Telescope and site models.
    label : str
        Instance label.
    bypass_optics : int or None
        Bypass_Optics value of the run (None: sim_telarray applies its optics to the photons).
    atmospheric_transmission : str or Path or None
        Zero-depth transmission table read by sim_telarray (None: sim_telarray applies the
        atmospheric extinction itself).
    efficiency_envelope : bool
        True if the photons are thinned by the quantum-efficiency envelope: sim_telarray then reads
        the QE table divided by the envelope (see ``simtools.simulation_software.efficiency``).
    photons_thinned : bool
        True if an upstream stage thins the photons (sim_telarray runs with min_photons disabled).
    """

    streams = True

    def __init__(
        self,
        simulation_parameters,
        array_model,
        label=None,
        context=None,
        bypass_optics=None,
        atmospheric_transmission=None,
        efficiency_envelope=False,
        photons_thinned=False,
    ):
        """Initialize SimTelArrayStage."""
        self.simulator_array = SimulatorArray(
            simulation_parameters=simulation_parameters,
            array_model=array_model,
            label=label,
            context=context,
        )
        self.array_model = array_model
        self.bypass_optics = bypass_optics
        self.atmospheric_transmission = atmospheric_transmission
        self.efficiency_envelope = efficiency_envelope
        if photons_thinned:
            self.simulator_array.min_photons = THINNED_MIN_PHOTONS

    @classmethod
    def instances(cls, inputs, label=None):
        """Return one sim_telarray stage per array model (one for a light-emission run)."""
        mode = inputs.options_of(cls.name)[BYPASS_OPTICS_OPTION]
        if mode == BYPASS_CAMERA_ONLY:
            _check_camera_only_build(inputs.context)
        zero_extinction = bool(inputs.chain.skipped_factors(cls.name))
        if inputs.calibration is not None:
            return _light_instances(cls, inputs, mode, zero_extinction)
        return _array_instances(cls, inputs, label, mode, zero_extinction)

    @property
    def file_list(self):
        """Files of this stage by file type."""
        return self.simulator_array.file_list

    @property
    def runner_service(self):
        """File naming of this stage."""
        return self.simulator_array.runner_service

    def prepare(self, run_number, input_file=None, output=None):
        """
        Write the sim_telarray configuration and return the run lines.

        Parameters
        ----------
        run_number : int
            Run number.
        input_file : str or Path
            CORSIKA output file, or ``STDIN`` when reading a stream.
        output : None
            sim_telarray writes its output file.

        Returns
        -------
        list[str]
            Shell lines running sim_telarray (one per run of a run set; one line when
            reading a stream).
        """
        if output is not None:
            raise ValueError("sim_telarray output can only be written to its file.")
        if self.efficiency_envelope:
            # read before the model directory: the scaled QE tables replace the model's QE tables
            self.simulator_array.include_directories = (self.write_scaled_tables(run_number),)
        command = self.simulator_array.make_run_command(
            run_number=run_number,
            input_file=input_file,
            bypass_optics=self.bypass_optics,
            atmospheric_transmission=self.atmospheric_transmission,
        )
        resources_file = self.runner_service.get_file_name(
            file_type="sim_telarray_resources", run_number=run_number
        )
        log_file = self.runner_service.get_file_name(
            file_type="sim_telarray_log", run_number=run_number
        )
        streamed = input_file == STDIN
        runs = 1 if streamed else self.simulator_array.runs_per_set
        resource_files = [
            resources_file.with_name(
                f"{resources_file.stem}.part{run_index + 1:04d}{resources_file.suffix}"
            )
            if runs > 1
            else resources_file
            for run_index in range(runs)
        ]
        lines = [
            f"{sim_telarray_env_as_string()} "
            + shlex.join(
                build_accounting_command(
                    command,
                    process_resources_file,
                    "sim_telarray",
                    run_number,
                    model_version=self.array_model.model_version,
                    log_file=log_file,
                    input_file=None if streamed else input_file,
                )
            )
            for process_resources_file in resource_files
        ]
        self.file_list["sim_telarray_resources"] = (
            resource_files[0] if len(resource_files) == 1 else resource_files
        )
        return lines

    def write_scaled_tables(self, run_number):
        """Write the QE / envelope table of each telescope type and return their directory."""
        directory = efficiency.envelope_directory(
            self.simulator_array.context, run_number, self.array_model.model_version
        )
        return efficiency.write_scaled_tables(
            directory / efficiency.SCALED_TABLE_DIRECTORY,
            efficiency.array_envelope(self.array_model),
        )

    def output_file(self, run_number):
        """Return the sim_telarray output file."""
        return self.runner_service.get_file_name("sim_telarray_output", run_number=run_number)

    def environment(self):
        """Return the sim_telarray environment variables."""
        environment = dict(SIM_TELARRAY_ENV)
        max_photoelectrons = self.simulator_array.context.args.get("max_photoelectrons")
        if max_photoelectrons is not None:
            environment["MAX_PHOTOELECTRONS"] = str(max_photoelectrons)
        return environment

    def get_resources(self, runtime=None):
        """Return computing resources used."""
        return self.runner_service.get_resources(runtime=runtime)


class LightSimTelArrayStage:
    """
    sim_telarray run on the photons of a calibration light source (one telescope).

    The command is that of ``SimulatorLightEmission.make_sim_telarray_command``. The photon
    file is handed over by the stage before it.

    Parameters
    ----------
    stage_class : type
        Stage class of the chain (name and products).
    calibration : SimulatorLightEmission
        Light-emission simulation of one telescope and one light source.
    bypass_optics : int or None
        Bypass_Optics value (None: derived by the light-emission simulation from the light source).
    atmospheric_transmission : str or Path or None
        Zero-depth transmission table read by sim_telarray (None: none).
    """

    streams = False

    def __init__(self, stage_class, calibration, bypass_optics=None, atmospheric_transmission=None):
        """Initialize LightSimTelArrayStage."""
        self.name = stage_class.name
        self.consumes = stage_class.consumes
        self.produces = stage_class.produces
        self.calibration = calibration
        self.bypass_optics = bypass_optics
        self.atmospheric_transmission = atmospheric_transmission
        self.file_list = {}

    def prepare(self, run_number, input_file=None, output=None):
        """
        Return the shell lines that run sim_telarray.

        Parameters
        ----------
        run_number : int
            Run number (the file names are those of the light-emission simulation).
        input_file : str or Path or None
            Photon file of the stage before (default: the IACT file of the LightEmission
            program).
        output : None
            sim_telarray writes its output file.

        Returns
        -------
        list of str
            Shell lines; the IACT file of the LightEmission program is removed afterwards.
        """
        del run_number
        if output is not None or input_file == STDIN:
            raise ValueError("sim_telarray of a light source reads a file and writes its file.")
        iact_output = self.calibration.runner_service.get_file_name(file_type="iact_output")
        command = self.calibration.make_sim_telarray_command(
            input_file=input_file,
            bypass_optics=self.bypass_optics,
            atmospheric_transmission=self.atmospheric_transmission,
        )
        lines = [command.rstrip("\n")]
        if input_file is None or str(input_file) == str(iact_output):
            lines.append(f"rm -f '{iact_output}'")
        return lines

    def output_file(self, run_number=None):
        """Return the sim_telarray output file."""
        del run_number
        return self.calibration.runner_service.get_file_name(file_type="sim_telarray_output")

    def environment(self):
        """Return environment variables needed to run the stage (set in the command)."""
        return {}

    def get_resources(self, runtime=None):
        """Return the run time."""
        return {"runtime": runtime}
