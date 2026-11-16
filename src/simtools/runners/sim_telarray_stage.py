"""sim_telarray as a simulation stage (telescope and camera response)."""

import logging
import os
import shlex

from simtools.job_execution.process_accounting import build_accounting_command
from simtools.ray_tracing.backends.interface import get_backend
from simtools.runners.chain_spec import (
    ATMOSPHERIC_EXTINCTION,
    MIRROR_REFLECTIVITY,
    QUANTUM_EFFICIENCY,
    TELESCOPE_TRANSMISSION,
    applied_factors_of,
)
from simtools.runners.shower_routing import ShowerRouting, route_threads
from simtools.runners.stage_types import CAMERA_EVENTS, CHERENKOV_PHOTONS, STDIN
from simtools.runners.stages import get_stage_class
from simtools.simtel.photon_bunch_file import bypass_optics_level
from simtools.simtel.sim_telarray_capabilities import check_bypass_optics
from simtools.simtel.simtel_runner import SIM_TELARRAY_ENV, sim_telarray_env_as_string
from simtools.simtel.simulator_array import SimulatorArray
from simtools.simulation_software import efficiency
from simtools.simulation_software.atmosphere import export_atmosphere_files, write_zero_transmission
from simtools.simulation_software.products import CAMERA_PHOTONS_IACT

#: Chain option: 0 (default) runs sim_telarray's optics on CORSIKA photons; 1 or 2 gives photons
#: at the camera entrance (Bypass_Optics from the mirror class of the telescopes); 3 is
#: Bypass_Optics=3 (only with a sim_telarray build that supports it, see sim_telarray_capabilities).
BYPASS_OPTICS_OPTION = "bypass_optics"
#: sim_telarray min_photons when an upstream stage thins photons. sim_telarray skips a telescope
#: event below min_photons, counted on the thinned photons, so the cut is disabled (the
#: min_photoelectrons cut still applies after the QE).
THINNED_MIN_PHOTONS = 0
_logger = logging.getLogger(__name__)
#: Output directory of the zero-depth transmission table (sim_telarray, bypass 1 or 2). Separate
#: from the atmosphere stage: exporting the site table here must not overwrite its table.
ZERO_TRANSMISSION_DIRECTORY = "atmosphere_zero"
_KNOWN_OPTIONS = (BYPASS_OPTICS_OPTION,)
_BYPASS_MODES = (0, 1, 2, 3)
_BYPASS_MODE_CAPABILITY_CHECK = 3
_DUAL_MIRROR_CLASS = 2


def _bypass_mode(options):
    """Return the bypass mode of the stage options (0 if not given)."""
    unknown = sorted(set(options) - set(_KNOWN_OPTIONS))
    if unknown:
        raise ValueError(
            f"Unknown sim_telarray stage option(s): {', '.join(unknown)}; "
            f"known options: {', '.join(_KNOWN_OPTIONS)}."
        )
    mode = options.get(BYPASS_OPTICS_OPTION, 0)
    if mode not in _BYPASS_MODES:
        raise ValueError(
            f"Invalid {BYPASS_OPTICS_OPTION} option: {mode!r}; choose from "
            f"{', '.join(str(value) for value in _BYPASS_MODES)}."
        )
    return mode


def _upstream_factors(inputs, stage_name):
    """Return the efficiency factors applied by the stages before ``stage_name`` in the chain."""
    names = list(inputs.stages)
    upstream = frozenset()
    for name in names[: names.index(stage_name)] if stage_name in names else names:
        upstream |= applied_factors_of(name, inputs.options_of(name), upstream)
    return upstream


def uses_zero_transmission(stage_names, options_of):
    """
    Return True if sim_telarray reads a zero-depth transmission table in the chain.

    That is the case for Bypass_Optics 1 or 2 with an upstream stage that applies the extinction.

    Parameters
    ----------
    stage_names : tuple of str
        Stages of the chain in order.
    options_of : callable
        Returns the options of a stage name.
    """
    upstream = frozenset()
    for name in stage_names:
        if name == SimTelArrayStage.name:
            return _bypass_mode(options_of(name)) in (1, 2) and ATMOSPHERIC_EXTINCTION in upstream
        upstream |= applied_factors_of(name, options_of(name), upstream)
    return False


def _zero_transmission(site_model, context):
    """Write the zero-depth transmission table of a site model; return its file."""
    directory = context.paths.get_output_directory(ZERO_TRANSMISSION_DIRECTORY)
    _profile, transmission = export_atmosphere_files(site_model, directory)
    return write_zero_transmission(transmission, directory)


def _bypass_level(mode, array_model):
    """
    Return sim_telarray's Bypass_Optics value for a bypass mode (None without bypass).

    Modes 1 and 2 take the level from the mirror class of the telescopes of the array, which
    must be the same for all telescopes (Bypass_Optics is one setting of a sim_telarray run).
    """
    if mode == 0:
        return None
    if mode == 3:
        return 3
    dual_mirror = {
        telescope.get_parameter_value("mirror_class") == _DUAL_MIRROR_CLASS
        for telescope in array_model.telescope_models.values()
    }
    if len(dual_mirror) != 1:
        raise ValueError(
            f"Bypass_Optics needs telescopes of one mirror class in an array; "
            f"array {array_model.model_version!r} mixes single and dual mirror telescopes."
        )
    return bypass_optics_level(dual_mirror.pop())


def _configure_shower_routing(stages, inputs, label):
    """Configure optional photon routing for the created stages."""
    if not inputs.shower_configurations:
        return
    mode = inputs.args.get("shower_route_mode", "stream")
    if mode not in ("stream", "disk"):
        raise ValueError(f"Invalid shower route mode: {mode}; choose from stream, disk")
    if not getattr(get_backend(context=inputs.context), "traces_shower_photons", False):
        return
    routes = sum(
        len({telescope.name for telescope in stage.array_model.telescope_models.values()})
        for stage in stages
    )
    threads = route_threads(
        inputs.args.get("shower_route_threads", 1),
        routes,
        sequential=inputs.args.get("sequential", False),
        usable_cores=os.process_cpu_count() or 1,
    )
    for stage in stages:
        stage.shower_routing = ShowerRouting(
            inputs.shower_configurations[0], inputs.context, label=label, threads=threads
        )
        stage.streams = mode == "stream"


class SimTelArrayStage:
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
        Bypass_Optics value of the run (None: sim_telarray applies its optics to CORSIKA photons).
    efficiency_envelope : bool
        True if the photons are thinned by the quantum-efficiency envelope: sim_telarray then reads
        the QE table divided by the envelope (see ``simtools.simulation_software.efficiency``).
    """

    name = "sim_telarray"
    consumes = (CHERENKOV_PHOTONS, CAMERA_PHOTONS_IACT)
    produces = CAMERA_EVENTS
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
        self.shower_routing = None

    @classmethod
    def applied_factors(cls, options, upstream=frozenset()):
        """
        Return the efficiency factors that sim_telarray applies to the photons it reads.

        Without bypass (mode 0), sim_telarray applies the atmospheric extinction, mirror
        reflectivity, telescope transmission and quantum efficiency to CORSIKA photons. With
        bypass modes 1 and 2, the photons are given at the camera entrance and sim_telarray
        applies its own atmospheric extinction and quantum efficiency; the mirror reflectivity and
        the telescope transmission are applied upstream. Mode 3 applies only the quantum
        efficiency.

        Parameters
        ----------
        options : dict
            Stage options of the chain spec (``bypass_optics``).

        Returns
        -------
        frozenset of str
            Efficiency factors applied by sim_telarray.
        """
        mode = _bypass_mode(options)
        if mode == 0:
            return frozenset(
                {
                    ATMOSPHERIC_EXTINCTION,
                    MIRROR_REFLECTIVITY,
                    TELESCOPE_TRANSMISSION,
                    QUANTUM_EFFICIENCY,
                }
            )
        if mode in (1, 2):
            if ATMOSPHERIC_EXTINCTION in upstream:
                # an upstream stage applies the extinction; sim_telarray reads a zero-depth table
                return frozenset({QUANTUM_EFFICIENCY})
            return frozenset({QUANTUM_EFFICIENCY, ATMOSPHERIC_EXTINCTION})
        return frozenset({QUANTUM_EFFICIENCY})

    @classmethod
    def instances(cls, inputs, label=None):
        """Return one sim_telarray stage per array model (one for a light-emission run)."""
        options = inputs.options_of(cls.name)
        mode = _bypass_mode(options)
        if mode == _BYPASS_MODE_CAPABILITY_CHECK:
            # Checked before any stage is built, so no job is written for an unsupported level.
            check_bypass_optics(mode, inputs.context)
        zero_extinction = mode in (1, 2) and ATMOSPHERIC_EXTINCTION in _upstream_factors(
            inputs, cls.name
        )
        if inputs.calibration is not None:
            return [
                LightSimTelArrayStage(
                    inputs.calibration,
                    _light_bypass_level(inputs, mode),
                    atmospheric_transmission=(
                        _zero_transmission(inputs.calibration.site_model, inputs.context)
                        if zero_extinction
                        else None
                    ),
                )
            ]
        envelope = efficiency.chain_envelope_enabled(inputs)
        thinned = efficiency.ATMOSPHERE_STAGE in inputs.stages
        if thinned:
            _logger.info(
                "Upstream %s thins photons: sim_telarray runs with min_photons=%s",
                efficiency.ATMOSPHERE_STAGE,
                THINNED_MIN_PHOTONS,
            )
        stages = [
            cls(
                simulation_parameters=parameters,
                array_model=array_model,
                label=label,
                context=inputs.context,
                bypass_optics=_bypass_level(mode, array_model),
                atmospheric_transmission=(
                    _zero_transmission(array_model.site_model, inputs.context)
                    if zero_extinction
                    else None
                ),
                efficiency_envelope=envelope,
                photons_thinned=thinned,
            )
            for parameters, array_model in zip(inputs.simulation_parameters, inputs.array_models)
        ]
        _configure_shower_routing(stages, inputs, label)
        return stages

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
        if self.shower_routing is not None:
            return self.shower_routing.commands(self.simulator_array, run_number, input_file)
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


def _light_bypass_level(inputs, mode):
    """
    Return the Bypass_Optics level of a light-emission run (None: derived by the simulation).

    Without the option ``bypass_optics``, the level follows the product of the stage before
    sim_telarray: photons of the LightEmission program need the level of the light source type;
    camera-frame photons need the option.
    """
    if BYPASS_OPTICS_OPTION in inputs.options_of(SimTelArrayStage.name):
        if mode in (1, 2):
            return bypass_optics_level(
                inputs.calibration.telescope_model.get_parameter_value("mirror_class")
                == _DUAL_MIRROR_CLASS
            )
        return mode
    position = inputs.stages.index(SimTelArrayStage.name) if inputs.stages else 0
    if position > 0 and get_stage_class(inputs.stages[position - 1]).produces == (
        CAMERA_PHOTONS_IACT
    ):
        raise ValueError(
            f"The option {BYPASS_OPTICS_OPTION!r} of stage {SimTelArrayStage.name!r} is required "
            f"for the camera-frame photons of {inputs.stages[position - 1]!r}: "
            f"use {SimTelArrayStage.name}:{BYPASS_OPTICS_OPTION}=1 (or 2 or 3)."
        )
    return None


class LightSimTelArrayStage:
    """
    sim_telarray run on the photons of a calibration light source (one telescope).

    The command is that of ``SimulatorLightEmission.make_sim_telarray_command``. The photon
    file is handed over by the stage before it.

    Parameters
    ----------
    calibration : SimulatorLightEmission
        Light-emission simulation of one telescope and one light source.
    bypass_optics : int or None
        Bypass_Optics value (0: none; None: derived by the light-emission simulation).
    """

    name = SimTelArrayStage.name
    consumes = SimTelArrayStage.consumes
    produces = CAMERA_EVENTS
    streams = False

    def __init__(self, calibration, bypass_optics=None, atmospheric_transmission=None):
        """Initialize LightSimTelArrayStage."""
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
