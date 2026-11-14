"""Atmosphere of the simulation chain as a simulation stage (obdeect propagation).

In a chain::

    simulation_chain: [corsika, corsika_to_stream, obdeect_atmosphere, obdeect_optics,
                       stream_to_simtel, sim_telarray]

The command is built in ``simtools.simulation_chain.obdeect.atmosphere``.
"""

import shlex

from simtools.runners.chain_spec import ATMOSPHERIC_EXTINCTION
from simtools.runners.stages import STDIN, StreamTo
from simtools.runtime.context import RunContext
from simtools.simulation_chain.atmosphere import (
    DEFAULT_MODE,
    DEFAULT_SEED,
    build_atmosphere_request,
)
from simtools.simulation_chain.obdeect.atmosphere import propagation_command
from simtools.simulation_chain.products import PHOTONS_GROUND, PHOTONS_SKY

STAGE_OPTIONS = ("seed", "mode", "wavelength_range_nm")
OUTPUT_DIRECTORY = "atmosphere"


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
    """

    name = "obdeect_atmosphere"
    consumes = (PHOTONS_SKY, PHOTONS_GROUND)
    produces = PHOTONS_GROUND
    streams = True

    def __init__(self, request, label=None, context=None):
        """Initialize ObdeectAtmosphereStage."""
        self.request = request
        self.label = label or self.name
        self.context = context or RunContext.from_environment()
        self.file_list = {}

    @classmethod
    def instances(cls, inputs, label=None):
        """
        Return one atmosphere stage per array model.

        Stage options (after the stage name in the chain): ``seed``, ``mode`` (``thinned`` or
        ``weighted``) and ``wavelength_range_nm`` (two values); unknown options are rejected.
        """
        options = inputs.options_of(cls.name)
        unknown = sorted(set(options) - set(STAGE_OPTIONS))
        if unknown:
            raise ValueError(
                f"Unknown options of stage {cls.name!r}: {unknown}; known: {list(STAGE_OPTIONS)}"
            )
        return [
            cls(
                build_atmosphere_request(
                    array_model.site_model,
                    array_model.telescope_models,
                    seed=options.get("seed", DEFAULT_SEED),
                    mode=options.get("mode", DEFAULT_MODE),
                    wavelength_range_nm=options.get("wavelength_range_nm"),
                ),
                label=label,
                context=inputs.context,
            )
            for array_model in inputs.array_models
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
        )
        line = shlex.join(command)
        return [line if downstream is None else f"{line} | {downstream}"]

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
    def applied_factors(cls, _options):
        """Return the efficiency factors applied by this stage."""
        return frozenset({ATMOSPHERIC_EXTINCTION})
