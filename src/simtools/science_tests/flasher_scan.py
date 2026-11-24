"""Expand a flasher science test from its simulation intensity settings."""

import copy

from simtools.utils import value_conversion

_RUN = "__FLASHER_RUN__"
_PHOTONS = "__FLASHER_PHOTONS__"


def _one_application(applications, name):
    matches = [app for app in applications if app.get("application") == name]
    if len(matches) != 1:
        raise ValueError(f"Flasher scan requires exactly one {name} template.")
    return matches[0]


def _levels(configuration):
    levels = value_conversion.parse_typed_sequence(configuration.get("flasher_photons"), int)
    if not levels or any(level <= 0 for level in levels) or len(set(levels)) != len(levels):
        raise ValueError("flasher_photons must contain distinct positive intensities.")
    events = value_conversion.parse_typed_sequence(configuration.get("number_of_events", 1), int)
    if len(events) not in (1, len(levels)) or any(count <= 0 for count in events):
        raise ValueError("number_of_events must contain one positive count or one per intensity.")
    if configuration.get("run_mode") != "full_simulation":
        raise ValueError("Flasher scan expansion requires full_simulation.")
    if len(levels) > 1 and configuration.get("max_workers", 1) < 2:
        raise ValueError("Flasher scan with multiple levels requires max_workers of at least 2.")
    return levels, events


def _expand_one(applications):
    """Generate analysis inputs and one plot per configured photon level.

    Parameters
    ----------
    applications : list[dict]
        Ordered simulation, analysis and plotting templates.

    Returns
    -------
    list[dict]
        Independent expanded application configurations.
    """
    applications = copy.deepcopy(applications)
    simulation = _one_application(applications, "simtools-simulate-flasher")
    analysis = _one_application(applications, "simtools-analyze-flasher-response")
    plot = _one_application(applications, "simtools-plot-simtel-events")
    if not applications.index(simulation) < applications.index(analysis) < applications.index(plot):
        raise ValueError("Flasher scan requires simulation, analysis, and plot in that order.")
    config = simulation["configuration"]
    levels, events = _levels(config)
    analysis_config = analysis["configuration"]
    plot_config = plot["configuration"]
    file_templates = analysis_config.get("simtel_files", [])
    if len(file_templates) != 1 or _RUN not in file_templates[0]:
        raise ValueError(
            "Flasher scan analysis requires one simtel_files template with __FLASHER_RUN__."
        )
    template = file_templates[0]
    if f"/run{_RUN}/" not in template:
        raise ValueError("Flasher scan file template must include the per-run directory.")
    output_path = config.get("output_path")
    if output_path and not template.startswith(f"{str(output_path).rstrip('/')}/"):
        raise ValueError("Flasher scan file template must be below the simulation output_path.")
    if "simtel_file" in plot_config or _PHOTONS not in plot_config.get("output_path", ""):
        raise ValueError(
            "Flasher scan plot must omit simtel_file and use __FLASHER_PHOTONS__ in output_path."
        )
    if len(levels) == 1:
        template = template.replace(f"/run{_RUN}/", "/")
    run_number = int(config.get("run_number", 1))
    if run_number < 1 or run_number + len(levels) - 1 > 999999:
        raise ValueError("Flasher scan run numbers must be between 1 and 999999.")
    files = [template.replace(_RUN, f"{run_number + index:06d}") for index in range(len(levels))]
    analysis_config.update(simtel_files=files, illumination=levels, events_per_level=events)
    plots = []
    for level, file in zip(levels, files, strict=True):
        item = copy.deepcopy(plot)
        item["configuration"]["simtel_file"] = file
        item["configuration"]["output_path"] = plot_config["output_path"].replace(
            _PHOTONS, str(level)
        )
        plots.append(item)
    index = applications.index(plot)
    applications[index : index + 1] = plots
    return applications


def _matching_applications(applications, simulation, telescope, names):
    """Find the ordered analysis and plot templates for a telescope."""
    matches = [simulation]
    for name in names[1:]:
        candidates = [
            app
            for app in applications
            if app.get("application") == name
            and app.get("configuration", {}).get("telescope") == telescope
        ]
        if len(candidates) != 1:
            raise ValueError(f"Flasher scan requires exactly one {name} for {telescope}.")
        matches.append(candidates[0])
    positions = [next(i for i, app in enumerate(applications) if app is item) for item in matches]
    if positions != sorted(positions):
        raise ValueError("Flasher scan requires simulation, analysis, and plot in that order.")
    return matches


def expand(applications):
    """Expand each telescope's flasher scan while preserving workflow step order.

    Parameters
    ----------
    applications : list[dict]
        Ordered telescope-expanded simulation, analysis and plotting templates.

    Returns
    -------
    list[dict]
        Independent expanded application configurations.
    """
    names = (
        "simtools-simulate-flasher",
        "simtools-analyze-flasher-response",
        "simtools-plot-simtel-events",
    )
    simulations = [app for app in applications if app.get("application") == names[0]]
    if len(simulations) <= 1:
        return _expand_one(applications)
    substitutions = {}
    used_telescopes = set()
    for simulation in simulations:
        telescopes = simulation.get("configuration", {}).get("telescopes", [])
        if len(telescopes) != 1 or telescopes[0] in used_telescopes:
            raise ValueError("Each flasher simulation must select one distinct telescope.")
        telescope = telescopes[0]
        used_telescopes.add(telescope)
        matches = _matching_applications(applications, simulation, telescope, names)
        expanded = _expand_one(matches)
        substitutions[id(matches[0])] = expanded[:1]
        substitutions[id(matches[1])] = expanded[1:2]
        substitutions[id(matches[2])] = expanded[2:]
    for app in applications:
        if app.get("application") in names and id(app) not in substitutions:
            raise ValueError("Every flasher analysis and plot must match a simulation telescope.")
    return [item for app in applications for item in substitutions.get(id(app), [app])]
