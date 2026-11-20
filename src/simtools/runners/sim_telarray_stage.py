"""sim_telarray as a simulation stage (telescope and camera response)."""

import logging
import os
import shlex

from simtools.job_execution.process_accounting import build_accounting_command
from simtools.ray_tracing.backends.interface import get_backend
from simtools.runners.shower_routing import ShowerRouting, route_threads
from simtools.runners.stage_types import STDIN
from simtools.runners.stages import (
    CatalogStage,
    applied_factors_of,
    efficiency_envelope_enabled,
    get_stage_class,
    stages_with,
)
from simtools.runtime import dependencies, software_catalog
from simtools.simtel.photon_bunch_file import bypass_optics_level
from simtools.simtel.simtel_runner import SIM_TELARRAY_ENV, sim_telarray_env_as_string
from simtools.simtel.simulator_array import SimulatorArray
from simtools.simulation_software import efficiency
from simtools.simulation_software.atmosphere import export_atmosphere_files, write_zero_transmission

#: Chain option: 0 (default) runs sim_telarray's optics on CORSIKA photons; 1 or 2 gives photons
#: at the camera entrance (Bypass_Optics from the mirror class of the telescopes); 3 is
#: Bypass_Optics=3 (requires a compatible sim_telarray build).
BYPASS_OPTICS_OPTION = "bypass_optics"
#: sim_telarray min_photons when an upstream stage thins photons. sim_telarray skips a telescope
#: event below min_photons, counted on the thinned photons, so the cut is disabled (the
#: min_photoelectrons cut still applies after the QE).
THINNED_MIN_PHOTONS = 0
_logger = logging.getLogger(__name__)
#: Output directory of the zero-depth transmission table (sim_telarray, bypass 1 or 2). Separate
#: from the atmosphere stage: exporting the site table here must not overwrite its table.
ZERO_TRANSMISSION_DIRECTORY = "atmosphere_zero"
_DUAL_MIRROR_CLASS = 2
_MODE_3_INCOMPATIBLE_TAGS = frozenset({"v2025-11-30-rc"})


def _bypass_mode(spec, options):
    """Return the bypass mode of the stage options (catalog default if not given)."""
    return spec.checked_options(options)[BYPASS_OPTICS_OPTION]


def _upstream_stages(inputs, stage_name):
    """Return the stages before ``stage_name`` in the chain (all stages if it is not listed)."""
    names = list(inputs.stages)
    return names[: names.index(stage_name)] if stage_name in names else names


def _upstream_factors(inputs, stage_name):
    """Return the efficiency factors applied by the stages before ``stage_name`` in the chain."""
    upstream = frozenset()
    for name in _upstream_stages(inputs, stage_name):
        upstream |= applied_factors_of(name, inputs.options_of(name), upstream)
    return upstream


def uses_zero_transmission(stage_names, options_of):
    """
    Return True if sim_telarray reads a zero-depth transmission table in the chain.

    That is the case when sim_telarray leaves a factor to an upstream stage
    (``skips_if_applied_upstream`` in the software catalog: the atmospheric extinction with
    Bypass_Optics 1 or 2).

    Parameters
    ----------
    stage_names : tuple of str
        Stages of the chain in order.
    options_of : callable
        Returns the options of a stage name.
    """
    upstream = frozenset()
    for name in stage_names:
        stage_class = get_stage_class(name)
        if issubclass(stage_class, SimTelArrayStage):
            return bool(stage_class.spec.skipped_factors(options_of(name), upstream))
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
        Bypass_Optics value of the run (None: sim_telarray applies its optics to CORSIKA photons).
    efficiency_envelope : bool
        True if the photons are thinned by the quantum-efficiency envelope: sim_telarray then reads
        the QE table divided by the envelope (see ``simtools.simulation_software.efficiency``).
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
        self.shower_routing = None

    @classmethod
    def instances(cls, inputs, label=None):
        """Return one sim_telarray stage per array model (one for a light-emission run)."""
        options = inputs.options_of(cls.name)
        mode = _bypass_mode(cls.spec, options)
        if mode == 3:
            _check_camera_only_build(inputs.context)
        zero_extinction = bool(
            cls.spec.skipped_factors(options, _upstream_factors(inputs, cls.name))
        )
        if inputs.calibration is not None:
            return [
                LightSimTelArrayStage(
                    cls,
                    inputs.calibration,
                    _light_bypass_level(cls, inputs, mode),
                    atmospheric_transmission=(
                        _zero_transmission(inputs.calibration.site_model, inputs.context)
                        if zero_extinction
                        else None
                    ),
                )
            ]
        envelope = efficiency_envelope_enabled(inputs)
        thinning = stages_with(_upstream_stages(inputs, cls.name), "thins_photons")
        if thinning:
            _logger.info(
                "Upstream %s thins photons: sim_telarray runs with min_photons=%s",
                ", ".join(thinning),
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
                photons_thinned=bool(thinning),
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


def _light_bypass_level(stage_class, inputs, mode):
    """
    Return the Bypass_Optics level of a light-emission run (None: derived by the simulation).

    Without the option ``bypass_optics``, the level follows the product of the stage before
    sim_telarray: photons of the LightEmission program need the level of the light source type;
    camera-frame photons need the option.
    """
    name = stage_class.name
    if BYPASS_OPTICS_OPTION in inputs.options_of(name):
        if mode in (1, 2):
            return bypass_optics_level(
                inputs.calibration.telescope_model.get_parameter_value("mirror_class")
                == _DUAL_MIRROR_CLASS
            )
        return mode
    position = inputs.stages.index(name) if name in inputs.stages else 0
    upstream = get_stage_class(inputs.stages[position - 1]).produces if position > 0 else None
    if upstream is not None and software_catalog.product(upstream).get("frame") == "camera":
        raise ValueError(
            f"The option {BYPASS_OPTICS_OPTION!r} of stage {name!r} is required "
            f"for the camera-frame photons of {inputs.stages[position - 1]!r}: "
            f"use {name}:{BYPASS_OPTICS_OPTION}=1 (or 2 or 3)."
        )
    return None


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
        Bypass_Optics value (0: none; None: derived by the light-emission simulation).
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
