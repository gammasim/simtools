"""Atmosphere stage of the obdeect chain: extinction and in-telescope path of the photons."""

import dataclasses
import shlex

import numpy as np

from simtools.runners.stages import (
    STDIN,
    CatalogStage,
    StreamTo,
    ends_camera_only,
    files_of_run,
    single_downstream,
    stage_file,
)
from simtools.runtime.context import RunContext
from simtools.simulation_software import efficiency, light_sources, optics
from simtools.simulation_software.atmosphere import (
    DEFAULT_SEED,
    build_atmosphere_request,
    write_obdeect_transmission,
)
from simtools.simulation_software.obdeect.atmosphere import propagation_command

OUTPUT_DIRECTORY = "atmosphere"


def _wavelength_range(inputs, options):
    """
    Return the wavelength range [nm] of photons without a wavelength (None: the run keys).

    A monochromatic source uses equal bounds for the envelope and efficiency status.
    """
    explicit = options.get("wavelength_range_nm")
    if explicit is not None or inputs.calibration is None:
        return explicit
    requested = light_sources.requested_wavelength(inputs.args.get("wavelength"))
    wavelength = light_sources.select_wavelength(inputs.calibration.calibration_model, requested)
    return (wavelength, wavelength)


def _simulation_seed(array_model):
    """Return the sim_telarray simulation seed of the array model (DEFAULT_SEED if unset)."""
    if array_model.sim_telarray_seed is None:
        return DEFAULT_SEED
    return optics.seeds_of(array_model.sim_telarray_seed)[1]


def run_seed(base_seed, run_number):
    """
    Return the thinning seed of one run, derived from a base seed and the run number.

    Runs with different run numbers or base seeds get independent random sequences; the same
    base seed and run number give the same seed.

    Parameters
    ----------
    base_seed : int
        Base seed (the sim_telarray simulation seed of the run).
    run_number : int
        Run number.

    Returns
    -------
    int
        Seed of the obdeect propagation.
    """
    return int(np.random.SeedSequence((int(base_seed), int(run_number))).generate_state(1)[0])


class ObdeectAtmosphereStage(CatalogStage):
    """
    Atmosphere stage: applies the atmospheric extinction and the in-telescope path to photons.

    The stage reads a photon stream in the sky or ground frame (stdin or file) and writes the
    photon stream in the ground frame. A streamed output is piped into the downstream command.

    Parameters
    ----------
    request : AtmosphereRequest
        Atmosphere inputs of one array.
    run_files : RunnerServices
        File naming of the run.
    context : RunContext or None
        Run context (output directory); default: the environment context.
    envelope_model : ArrayModel or None
        Array model whose telescope types bound the efficiency envelope; None: no envelope.
    seed_per_run : bool
        Derive the seed of each run from the request seed and the run number.
    """

    streams = True

    @classmethod
    def efficiency_envelope(cls, inputs):
        """
        Return True if the stage thins the photons by the efficiency envelope.

        The option ``efficiency_envelope`` decides. Without it, the envelope is on for a chain
        whose last stage simulates the camera only, and off otherwise. Calibration chains have
        no envelope by default and reject an explicit ``qe``.
        """
        options = inputs.options_of(cls.name)
        calibration = inputs.calibration is not None
        camera_only = not calibration and ends_camera_only(inputs)
        default = efficiency.ENVELOPE_ON if camera_only else efficiency.ENVELOPE_OFF
        enabled = efficiency.envelope_setting(options, default=default)
        if calibration and enabled:
            raise ValueError("The QE efficiency envelope is not supported for calibration chains.")
        return enabled

    def __init__(
        self, request, run_files=None, context=None, envelope_model=None, seed_per_run=False
    ):
        """Initialize ObdeectAtmosphereStage."""
        self.request = request
        self.seed_per_run = seed_per_run
        self.run_files = run_files
        self.context = context or RunContext.from_environment()
        self.envelope_model = envelope_model
        self.file_list = {}

    @classmethod
    def instances(cls, inputs, label=None):
        """
        Return the atmosphere stage of the array model.

        The options are declared in the software catalog. Without ``seed``, the seed of each run
        is derived from the simulation seed and the run number.
        """
        options = inputs.options_of(cls.name)
        array_model = inputs.array_models[0]
        explicit_seed = options.get("seed")
        request = build_atmosphere_request(
            array_model.site_model,
            array_model.telescope_models,
            seed=(_simulation_seed(array_model) if explicit_seed is None else explicit_seed),
            mode=options["mode"],
            wavelength_range_nm=_wavelength_range(inputs, options),
        )
        request = dataclasses.replace(
            request,
            transmission_file=write_obdeect_transmission(
                request.transmission_file,
                inputs.context.paths.get_output_directory(OUTPUT_DIRECTORY),
            ),
        )
        return [
            cls(
                request,
                run_files=files_of_run(inputs, label, required=False),
                context=inputs.context,
                envelope_model=array_model if cls.efficiency_envelope(inputs) else None,
                seed_per_run=explicit_seed is None,
            )
        ]

    def prepare(self, run_number, input_file=None, output=None):
        """
        Return the shell line that runs the atmosphere propagation.

        Parameters
        ----------
        run_number : int
            Run number.
        input_file : str or Path
            Input photon stream file, or ``STDIN`` when reading a stream.
        output : StreamTo or None
            Downstream command receiving the output stream; None writes the output file.

        Returns
        -------
        list[str]
            One shell line.
        """
        if input_file is None:
            raise ValueError(f"Stage {self.name!r} needs an input photon stream.")
        downstream = None
        if isinstance(output, StreamTo):
            downstream = single_downstream(output, self.name)
            output_file = STDIN
        else:
            output_file = self.output_file(run_number)
        request = self.request
        if self.seed_per_run:
            request = dataclasses.replace(request, seed=run_seed(request.seed, run_number))
        command = propagation_command(
            request,
            input_file=input_file,
            output_file=output_file,
            envelope_file=None if self.envelope_model is None else self.write_envelope(run_number),
        )
        line = shlex.join(command)
        return [line if downstream is None else f"{line} | {downstream}"]

    def write_envelope(self, run_number):
        """Write the quantum-efficiency bound of the array and return its file."""
        directory = efficiency.envelope_directory(
            self.context, run_number, self.envelope_model.model_version
        )
        bound = efficiency.array_envelope(self.envelope_model).bound
        return efficiency.write_envelope(directory / efficiency.ENVELOPE_FILE_NAME, bound)

    def output_file(self, run_number):
        """Return the photon stream file written when the output is not streamed."""
        return stage_file(self, "output", run_number)
