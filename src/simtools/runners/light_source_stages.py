"""Calibration light sources as photon-stream stages and sim_telarray LightEmission stages."""

import dataclasses
import json
import shlex
from pathlib import Path

from simtools.model.model_utils import (
    initialize_simulation_models,
    read_overwrite_model_parameter_dict,
)
from simtools.runners.stages import (
    CatalogStage,
    StreamTo,
    files_of_run,
    single_downstream,
    stage_file,
)
from simtools.runtime import software_catalog
from simtools.simtel.simtel_seeds import instance_seed_pairs
from simtools.simulation_software import light_sources
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


def _file_token(telescope, light_source, wavelength_nm):
    """Return the file-name suffix of a stage instance."""
    text = f"{telescope}_{light_source}_{wavelength_nm:.0f}nm"
    return "_" + "".join(c if (c.isalnum() or c in "-_") else "_" for c in text)


class LightSourceStage(CatalogStage):
    """
    Light-source stage: photons of one calibration device seen by one telescope.

    The light-source type is the catalog field ``light_source`` of the stage (see
    ``light_sources`` in ``resources/simulation_software.yml``).

    Parameters
    ----------
    request : FlasherRequest or IlluminatorRequest
        Request of the source type.
    work_directory : str or Path
        Directory of the tables of the source.
    file_token : str
        Suffix of the table file names of this instance.
    executable : str, optional
        Program of the source type (default: the one of the source type).
    simulation_models_path : str or Path, optional
        Simulation models (for sources that read model files).
    run_files : RunnerServices, optional
        File naming of the run (output file when the output is not streamed).
    context : RunContext, optional
        Run context (model reader of the simulation models, used by sources that read them).
    """

    streams = True

    def __init__(
        self,
        request,
        work_directory,
        file_token="",
        executable=None,
        simulation_models_path=None,
        run_files=None,
        context=None,
    ):
        """Initialize LightSourceStage."""
        self.run_files = run_files
        self.request = request
        self.work_directory = Path(work_directory)
        self.file_token = file_token
        self.executable = executable
        self.simulation_models_path = simulation_models_path
        self.file_list = None
        self._context = context

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
        if light_sources.source_name_for_model(model_type) != cls.spec.light_source:
            raise ValueError(
                f"Light source {config.get('light_source')!r} is of type {model_type!r}; "
                f"stage {cls.name!r} handles {cls.spec.light_source!r}."
            )
        config["wavelength"] = light_sources.requested_wavelength(config.get("wavelength"))
        instrument_seed, simulation_seed = instance_seed_pairs(context=context)[0]
        config |= {"telescope": telescope, "instrument_seed": instrument_seed}
        request = light_sources.build_request(
            cls.spec.light_source,
            calibration_model,
            telescope_model,
            site_model,
            config,
            seed=simulation_seed,
        )
        token = _file_token(telescope, config.get("light_source"), request.wavelength_nm)
        work_directory = context.paths.get_output_directory(LIGHT_EMISSION_DIRECTORY)
        stage = cls(
            request,
            work_directory,
            token,
            executable=config.get("executable"),
            simulation_models_path=config.get("simulation_models_path"),
            run_files=files_of_run(inputs, label, required=False),
            context=context,
        )
        return [stage]

    def output_file(self, run_number):
        """Return the photon-stream file written when the output is not streamed."""
        return stage_file(self, "output", run_number)

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
        downstream = single_downstream(output, self.name) if streamed else None
        target = None if streamed else self.output_file(run_number)
        self.work_directory.mkdir(parents=True, exist_ok=True)
        command = shlex.join(
            light_sources.source_command(
                self.spec.light_source,
                request,
                self.work_directory,
                output=target,
                file_token=self.file_token,
                executable=self.executable,
                simulation_models_path=self.simulation_models_path,
            )
        )
        model_context = software_catalog.light_source(self.spec.light_source).get("model_context")
        if model_context and self._context is not None:
            source = getattr(self._context.model_reader, "source_config", None)
            overrides = read_overwrite_model_parameter_dict(context=self._context)
            if isinstance(source, dict) or overrides:
                path = self.work_directory / f"flasher_model_context{self.file_token}.json"
                path.write_text(
                    json.dumps({"model_source": source, "overwrite_model_parameters": overrides}),
                    encoding="utf-8",
                )
                command = f"env {shlex.quote(f'SIMTOOLS_FLASHER_CONTEXT={path}')} {command}"
        return [f"{command} | {downstream}"] if streamed else [command]

    def get_resources(self, runtime=None):
        """Return the run time and the number of simulated events."""
        events = self.request.events
        total = sum(events) if isinstance(events, tuple | list) else events
        return {"runtime": runtime, "n_events": int(total)}


class LightEmissionStage(CatalogStage):
    """
    LightEmission program of sim_telarray as the first stage of a chain.

    The stage runs the program of the light source type of a ``SimulatorLightEmission`` and
    checks the IACT file it writes, which is handed over to the next stage.

    Parameters
    ----------
    calibration : SimulatorLightEmission
        Light-emission simulation of one telescope and one light source.
    """

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

    def get_resources(self, runtime=None):
        """Return the run time."""
        return {"runtime": runtime}
