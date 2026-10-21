"""Select software-specific configuration writers for resolved simulation models."""


def _simtel_writer(model):
    """Construct the sim_telarray model writer."""
    # pylint: disable-next=import-outside-toplevel
    from simtools.simtel.model_writer import SimtelModelWriter

    return SimtelModelWriter(model)


_WRITERS = {"sim_telarray": _simtel_writer}


def register_model_writer(simulation_software, factory):
    """Register configuration writing for a telescope simulation program.

    Parameters
    ----------
    simulation_software : str
        Unique software name.
    factory : callable
        Callable accepting a resolved model and returning a writer. Writers
        implement ``write_config_file(additional_models=None, label=None)``
        for individual models and ``export_config_files()`` for arrays.

    Raises
    ------
    ValueError
        If the name is invalid, already registered, or the factory is not callable.
    """
    _register_writer(_WRITERS, simulation_software, factory)


def available_model_writers():
    """Return registered model configuration writers."""
    return tuple(sorted(_WRITERS))


def get_model_writer(model, simulation_software="sim_telarray"):
    """Return the selected model configuration writer, retaining its table cache.

    Parameters
    ----------
    model : ModelParameter or ArrayModel
        Resolved simulation model.
    simulation_software : str
        Registered configuration writer.

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
        model.configuration_writers[simulation_software] = factory(model)
    return model.configuration_writers[simulation_software]


def _corsika_configuration(array_model, run_number, label=None):
    """Construct the CORSIKA7 configuration writer."""
    # pylint: disable-next=import-outside-toplevel
    from simtools.corsika.corsika_config import CorsikaConfig

    return CorsikaConfig(array_model=array_model, run_number=run_number, label=label)


def _light_emission_writer(simulation):
    """Construct the current calibration-light configuration writer."""
    # pylint: disable-next=import-outside-toplevel
    from simtools.simtel.light_emission_config_writer import LightEmissionConfigWriter

    return LightEmissionConfigWriter(simulation)


_SHOWER_WRITERS = {"corsika": _corsika_configuration}
_LIGHT_SOURCE_WRITERS = {"light_emission": _light_emission_writer}


def register_shower_writer(simulation_software, factory):
    """Register a shower configuration writer.

    Parameters
    ----------
    simulation_software : str
        Unique shower-software name.
    factory : callable
        Accepts ``array_model``, ``run_number`` and ``label``; returns a writer
        exposing shared ``simulation_parameters`` and its native configuration.
    """
    _register_writer(_SHOWER_WRITERS, simulation_software, factory)


def register_light_source_writer(simulation_software, factory):
    """Register configuration writing for a calibration light source.

    Parameters
    ----------
    simulation_software : str
        Unique light-source software name.
    factory : callable
        Accepts the physical light-source simulation setup and returns a writer
        implementing ``make_command(iact_output)``.
    """
    _register_writer(_LIGHT_SOURCE_WRITERS, simulation_software, factory)


def _register_writer(writers, simulation_software, factory):
    """Validate and register a software-specific writer factory."""
    if not simulation_software or not callable(factory):
        raise ValueError("A software name and callable configuration-writer factory are required.")
    if simulation_software in writers:
        raise ValueError(f"Configuration writer '{simulation_software}' is already registered.")
    writers[simulation_software] = factory


def get_shower_configuration(array_model, run_number, label=None, simulation_software="corsika"):
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
    return factory(array_model=array_model, run_number=run_number, label=label)


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
