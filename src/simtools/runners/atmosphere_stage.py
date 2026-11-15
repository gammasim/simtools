"""Apply obdeect atmospheric extinction to a photon stream."""

import dataclasses
import shlex

import astropy.units as u

from simtools.runners.chain_spec import ATMOSPHERIC_EXTINCTION
from simtools.runners.stages import STDIN, StreamTo
from simtools.runtime.context import RunContext
from simtools.simulation_software import efficiency, light_sources
from simtools.simulation_software.atmosphere import (
    DEFAULT_MODE,
    DEFAULT_SEED,
    build_atmosphere_request,
    write_obdeect_transmission,
)
from simtools.simulation_software.obdeect.atmosphere import propagation_command
from simtools.simulation_software.products import PHOTONS_GROUND, PHOTONS_SKY
from simtools.utils import value_conversion

#: Width [nm] added above a monochromatic light-source wavelength (see ``_wavelength_range``).
MONOCHROMATIC_BAND_NM = 1.0
STAGE_OPTIONS = ("seed", "mode", "wavelength_range_nm", efficiency.ENVELOPE_OPTION)
OUTPUT_DIRECTORY = "atmosphere"


def _requested_wavelength(value):
    """Convert a configured wavelength to a quantity; numeric values are in nm."""
    if value is None:
        return None
    values = value_conversion.ensure_list(value)
    if len(values) != 1:
        raise ValueError(f"Expected exactly one wavelength; got {values}.")
    wavelength = values[0]
    return wavelength if hasattr(wavelength, "to") else float(wavelength) * u.nm


def _wavelength_range(inputs, options):
    """
    Return the wavelength range [nm] of photons without a wavelength (None: the run keys).

    obdeect needs lo < hi in every run. A monochromatic light source has lo == hi in its run
    keys, so an illuminator chain gets its source wavelength with a 1 nm upper edge. Photons of
    the source carry their wavelength already, so none is drawn from this range: it only sets the
    envelope grid and the efficiency-status records of the run.
    """
    explicit = options.get("wavelength_range_nm")
    if explicit is not None or inputs.calibration is None:
        return explicit
    requested = _requested_wavelength(inputs.args.get("wavelength"))
    wavelength = light_sources.select_wavelength(inputs.calibration.calibration_model, requested)
    return (wavelength, wavelength + MONOCHROMATIC_BAND_NM)


class ObdeectAtmosphereStage:
    """
    Atmosphere stage: applies the atmospheric extinction and the in-telescope path to photons.

    The stage reads a photon stream in the sky or ground frame (stdin or file) and writes the
    photon stream in the ground frame. A streamed output is piped into the downstream command.

    Parameters
    ----------
    request : AtmosphereRequest
        Atmosphere inputs of one array.
    label : str
        Instance label.
    context : RunContext or None
        Run context (output directory); default: the environment context.
    envelope_model : ArrayModel or None
        Array model whose telescope types bound the efficiency envelope; None: no envelope.
    """

    name = "obdeect_atmosphere"
    consumes = (PHOTONS_SKY, PHOTONS_GROUND)
    produces = PHOTONS_GROUND
    streams = True

    def __init__(self, request, label=None, context=None, envelope_model=None):
        """Initialize ObdeectAtmosphereStage."""
        self.request = request
        self.label = label or self.name
        self.context = context or RunContext.from_environment()
        self.envelope_model = envelope_model
        self.file_list = {}

    @classmethod
    def instances(cls, inputs, label=None):
        """
        Return one atmosphere stage per array model.

        Stage options (after the stage name in the chain): ``seed``, ``mode`` (``thinned`` or
        ``weighted``), ``wavelength_range_nm`` (two values) and ``efficiency_envelope`` (``qe``,
        default: thin photons by the quantum-efficiency bound of the telescopes; ``none``).
        Unknown options are rejected.
        """
        options = inputs.options_of(cls.name)
        unknown = sorted(set(options) - set(STAGE_OPTIONS))
        if unknown:
            raise ValueError(
                f"Unknown options of stage {cls.name!r}: {unknown}; known: {list(STAGE_OPTIONS)}"
            )
        envelope = efficiency.envelope_for_inputs(inputs)
        wavelength_range = _wavelength_range(inputs, options)
        stage_list = []
        for array_model in inputs.array_models:
            request = build_atmosphere_request(
                array_model.site_model,
                array_model.telescope_models,
                seed=options.get("seed", DEFAULT_SEED),
                mode=options.get("mode", DEFAULT_MODE),
                wavelength_range_nm=wavelength_range,
            )
            request = dataclasses.replace(
                request,
                transmission_file=write_obdeect_transmission(
                    request.transmission_file,
                    inputs.context.paths.get_output_directory(OUTPUT_DIRECTORY),
                ),
            )
            stage_list.append(
                cls(
                    request,
                    label=label,
                    context=inputs.context,
                    envelope_model=array_model if envelope else None,
                )
            )
        return stage_list

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
            if len(output.commands) != 1:
                raise ValueError(
                    f"Stage {self.name!r} pipes into one downstream command, "
                    f"got {len(output.commands)}."
                )
            downstream = output.commands[0]
            output_file = STDIN
        else:
            output_file = self.output_file(run_number)
        command = propagation_command(
            self.request,
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
        directory = self.context.paths.get_output_directory(OUTPUT_DIRECTORY)
        return directory / f"{self.label}-run{run_number:06d}.pst"

    def environment(self):
        """Return the environment variables needed (none)."""
        return {}

    def get_resources(self, runtime=None):  # pylint: disable=unused-argument
        """Return computing resources used (none: the chain's first stage reports them)."""
        return {}

    @classmethod
    def applied_factors(cls, _options, upstream=frozenset()):  # pylint: disable=unused-argument
        """Return the efficiency factors applied by this stage: the atmospheric extinction."""
        return frozenset({ATMOSPHERIC_EXTINCTION})
