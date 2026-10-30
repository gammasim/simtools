"""Select software-specific configuration writers for resolved simulation models."""


def _simtel_writer(model, context=None):
    """Construct the sim_telarray model writer."""
    # pylint: disable-next=import-outside-toplevel
    from simtools.simtel.model_writer import SimtelModelWriter

    return SimtelModelWriter(model, context=context)


_WRITERS = {"sim_telarray": _simtel_writer}


def export_model_files(model, destination=None, update_if_necessary=False):
    """Export the parameter files of a model into its model directory.

    Parameters
    ----------
    model : ModelParameter
        Model whose parameter files are exported.
    destination : pathlib.Path, optional
        Target directory. Defaults to the model configuration directory.
    update_if_necessary : bool
        Skip the export if the model files were already exported and not changed since.
    """
    if update_if_necessary and model.model_files_exported:
        return
    model.model_reader.export_model_files(
        parameters=model.parameters,
        dest=destination or model.config_file_directory,
    )
    model.model_files_exported = True


def get_model_writer(model, simulation_software="sim_telarray", context=None):
    """Return the selected model configuration writer, retaining its table cache.

    Parameters
    ----------
    model : ModelParameter or ArrayModel
        Resolved simulation model.
    simulation_software : str
        Registered configuration writer.
    context : RunContext, optional
        Run context; the model's own context is used if not given.

    Returns
    -------
    object
        Software-specific model writer.

    Raises
    ------
    ValueError
        If the selected software is not registered.
    """
    try:
        factory = _WRITERS[simulation_software]
    except KeyError as exc:
        raise ValueError(f"Unknown model configuration writer '{simulation_software}'.") from exc
    if simulation_software not in model.configuration_writers:
        model.configuration_writers[simulation_software] = factory(model, context=context)
    return model.configuration_writers[simulation_software]


def _corsika_configuration(array_model, run_number, label=None, context=None):
    """Construct the CORSIKA7 configuration writer."""
    # pylint: disable-next=import-outside-toplevel
    from simtools.corsika.corsika_config import CorsikaConfig

    return CorsikaConfig(
        array_model=array_model, run_number=run_number, label=label, context=context
    )


def _light_emission_writer(simulation):
    """Construct the current calibration-light configuration writer."""
    # pylint: disable-next=import-outside-toplevel
    from simtools.simtel.light_emission_config_writer import LightEmissionConfigWriter

    return LightEmissionConfigWriter(simulation)


_SHOWER_WRITERS = {"corsika": _corsika_configuration}
_LIGHT_SOURCE_WRITERS = {"light_emission": _light_emission_writer}


def get_shower_configuration(
    array_model, run_number, label=None, simulation_software="corsika", context=None
):
    """Create a shower configuration for the selected software.

    Parameters
    ----------
    array_model : ArrayModel
        Resolved array and site model.
    run_number : int
        Run identifier.
    label : str, optional
        File naming label.
    simulation_software : str
        Registered shower configuration writer.

    Returns
    -------
    object
        Shower configuration with common simulation parameters.
    """
    try:
        factory = _SHOWER_WRITERS[simulation_software]
    except KeyError as exc:
        raise ValueError(f"Unknown shower configuration writer '{simulation_software}'.") from exc
    return factory(array_model=array_model, run_number=run_number, label=label, context=context)


def get_light_source_writer(simulation):
    """Return the configuration writer for a calibration-source simulation.

    Parameters
    ----------
    simulation : SimulatorLightEmission
        Physical setup with ``light_emission_config`` and resolved models.

    Returns
    -------
    object
        Selected calibration-source configuration writer.
    """
    software = simulation.light_emission_config.get("light_source_software", "light_emission")
    try:
        factory = _LIGHT_SOURCE_WRITERS[software]
    except KeyError as exc:
        raise ValueError(f"Unknown light-source configuration writer '{software}'.") from exc
    if not hasattr(simulation, "configuration_writers"):
        simulation.configuration_writers = {}
    if software not in simulation.configuration_writers:
        simulation.configuration_writers[software] = factory(simulation)
    return simulation.configuration_writers[software]
