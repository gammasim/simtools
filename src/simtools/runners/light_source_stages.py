"""Calibration light sources as photon-stream stages and sim_telarray LightEmission stages."""

import dataclasses
import shlex
from pathlib import Path

import astropy.units as u

from simtools.model.model_utils import initialize_simulation_models
from simtools.runners.stages import CHERENKOV_PHOTONS, StreamTo
from simtools.simtel.simtel_seeds import instance_seed_pairs
from simtools.simulation_software import light_sources
from simtools.simulation_software.products import PHOTONS_CAMERA, PHOTONS_SKY
from simtools.utils import value_conversion

LIGHT_EMISSION_DIRECTORY = "light_emission"


def _single(value, name):
    """Return the only value of a setting given as a value or a list."""
    values = value_conversion.ensure_list(value)
    if len(values) != 1:
        raise ValueError(f"Light-source stages take exactly one {name}; got {values}.")
    return values[0]


def _telescope(config):
    """Return the telescope name of the configuration."""
    if config.get("telescope"):
        return config["telescope"]
    return _single(config.get("telescopes"), "telescope")


def _wavelength(config):
    """Return the requested wavelength (quantity) or None; a number is in nm."""
    wavelength = config.get("wavelength")
    if wavelength is None:
        return None
    wavelength = _single(wavelength, "wavelength")
    return wavelength if hasattr(wavelength, "to") else float(wavelength) * u.nm


def _file_token(telescope, light_source, wavelength_nm):
    """Return the file-name suffix of a stage instance."""
    text = f"{telescope}_{light_source}_{wavelength_nm:.0f}nm"
    return "_" + "".join(c if (c.isalnum() or c in "-_") else "_" for c in text)


class LightSourceStage:
    """
    Light-source stage: photons of one calibration device seen by one telescope.

    Parameters
    ----------
    source_type : type
        Source-type class (see ``simtools.simulation_software.light_sources``).
    request : FlasherRequest or IlluminatorRequest
        Request of the source type.
    work_directory : str or Path
        Directory of the output file and the tables of the source.
    file_token : str
        Suffix of the file names of this instance.
    executable : str, optional
        Program of the source type (default: the one of the source type).
    """

    name = None
    source_name = None
    consumes = None
    produces = None
    streams = True

    def __init__(self, request, work_directory, file_token="", executable=None):
        """Initialize LightSourceStage."""
        self.request = request
        self.work_directory = Path(work_directory)
        self.file_token = file_token
        self.executable = executable
        self.file_list = None

    @classmethod
    def instances(cls, inputs, label=None):
        """
        Return the stage of a run (one telescope and one light source).

        Parameters
        ----------
        inputs : StageInputs
            Run inputs; the arguments and the stage options define the light-emission run.
        label : str, optional
            Label of the models.

        Returns
        -------
        list of LightSourceStage
            One stage.
        """
        config = dict(inputs.args) | inputs.options_of(cls.name)
        telescope = _telescope(config)
        context = inputs.context
        telescope_model, site_model, calibration_model = initialize_simulation_models(
            label=label,
            model_version=config.get("model_version"),
            site=config.get("site"),
            telescope_name=telescope,
            calibration_device_name=config.get("light_source"),
            calibration_device_type=config.get("light_source_type"),
            model_reader=getattr(context, "model_reader", None),
            model_directory=config.get("model_directory"),
            context=context,
        )
        model_type = str(calibration_model.get_parameter_value("flasher_type")).lower()
        if light_sources.source_name_for_model(model_type) != cls.source_name:
            raise ValueError(
                f"Light source {config.get('light_source')!r} is of type {model_type!r}; "
                f"stage {cls.name!r} handles {cls.source_name!r}."
            )
        config["wavelength"] = _wavelength(config)
        instrument_seed, simulation_seed = instance_seed_pairs(context=context)[0]
        config |= {"telescope": telescope, "instrument_seed": instrument_seed}
        request = light_sources.build_request(
            cls.source_name,
            calibration_model,
            telescope_model,
            site_model,
            config,
            seed=simulation_seed,
        )
        token = _file_token(telescope, config.get("light_source"), request.wavelength_nm)
        work_directory = context.paths.get_output_directory(LIGHT_EMISSION_DIRECTORY)
        return [
            cls(
                request,
                work_directory,
                token,
                executable=config.get("executable"),
            )
        ]

    def output_file(self, run_number):
        """Return the photon-stream file written when the output is not streamed."""
        return self.work_directory / f"{self.name}{self.file_token}_run{run_number:06d}.pst"

    def prepare(self, run_number, input_file=None, output=None):
        """
        Return the shell lines that make the photons.

        Parameters
        ----------
        run_number : int
            Run number.
        input_file : None
            A light source reads no input product.
        output : StreamTo or None
            One downstream command receiving the photon stream; None writes the stream file
            (see ``output_file``).

        Returns
        -------
        list of str
            Shell lines.
        """
        if input_file is not None:
            raise ValueError("A light source does not read an input product or stream.")
        request = dataclasses.replace(self.request, run_number=run_number)
        streamed = isinstance(output, StreamTo)
        if streamed and len(output.commands) != 1:
            raise ValueError(
                f"Stage {self.name!r} streams to exactly one downstream command; "
                f"got {len(output.commands)}."
            )
        target = None if streamed else self.output_file(run_number)
        self.work_directory.mkdir(parents=True, exist_ok=True)
        command = shlex.join(
            light_sources.source_command(
                self.source_name,
                request,
                self.work_directory,
                output=target,
                file_token=self.file_token,
                executable=self.executable,
            )
        )
        return [f"{command} | {output.commands[0]}"] if streamed else [command]

    def environment(self):
        """Return environment variables needed to run the stage (none)."""
        return {}

    def get_resources(self, runtime=None):
        """Return the run time and the number of simulated events."""
        events = self.request.events
        total = sum(events) if isinstance(events, tuple | list) else events
        return {"runtime": runtime, "n_events": int(total)}

    @staticmethod
    def applied_factors(options=None):
        """Return the efficiency factors applied to the photons (none)."""
        del options
        return frozenset()


class FlasherSourceStage(LightSourceStage):
    """Flasher (flat-fielding light source): photons at the camera plane of the telescope."""

    name = "obdeect_flasher"
    source_name = "flasher"
    produces = PHOTONS_CAMERA


class IlluminatorSourceStage(LightSourceStage):
    """Illuminator (external light source): photons in the sky frame."""

    name = "obdeect_illuminator"
    source_name = "illuminator"
    produces = PHOTONS_SKY


class LightEmissionStage:
    """
    LightEmission program of sim_telarray as the first stage of a chain.

    The stage runs the program of the light source type of a ``SimulatorLightEmission`` and
    checks the IACT file it writes, which is handed over to the next stage.

    Parameters
    ----------
    calibration : SimulatorLightEmission
        Light-emission simulation of one telescope and one light source.
    """

    name = "light_emission"
    consumes = None
    produces = CHERENKOV_PHOTONS
    streams = False

    def __init__(self, calibration):
        """Initialize LightEmissionStage."""
        self.calibration = calibration
        self.file_list = {}

    @classmethod
    def instances(cls, inputs, label=None):
        """Return the stage of the light-emission simulation of the run."""
        del label
        if inputs.calibration is None:
            raise ValueError(
                f"Stage {cls.name!r} needs a light-emission simulation "
                "(simtools-simulate-flasher or simtools-simulate-illuminator)."
            )
        return [cls(inputs.calibration)]

    def prepare(self, run_number, input_file=None, output=None):
        """
        Return the shell lines that run the LightEmission program and check its output.

        Parameters
        ----------
        run_number : int
            Run number (the file names are those of the light-emission simulation).
        input_file : None
            The program reads no input product.
        output : None
            The program writes the IACT file (see ``output_file``).

        Returns
        -------
        list of str
            Shell lines.
        """
        del run_number
        if input_file is not None or output is not None:
            raise ValueError(
                f"Stage {self.name!r} reads no input and writes its IACT file; "
                "it cannot be connected by a stream."
            )
        iact_output = self.output_file()
        return [
            self.calibration.light_emission_command(iact_output),
            self.calibration.iact_file_check(iact_output),
        ]

    def output_file(self, run_number=None):
        """Return the IACT file written by the LightEmission program."""
        del run_number
        return self.calibration.runner_service.get_file_name(file_type="iact_output")

    def environment(self):
        """Return environment variables needed to run the stage (none)."""
        return {}

    def get_resources(self, runtime=None):
        """Return the run time."""
        return {"runtime": runtime}

    @staticmethod
    def applied_factors(options=None):
        """Return the efficiency factors applied to the photons (none)."""
        del options
        return frozenset()
