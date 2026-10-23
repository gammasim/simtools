"""Run and export the standard camera-efficiency calculations."""

import logging
from pathlib import Path

from simtools.camera.camera_efficiency import CameraEfficiency
from simtools.io.ascii_handler import write_data_to_file
from simtools.model.model_utils import initialize_simulation_models
from simtools.utils import names
from simtools.visualization.matplotlib_backend import pyplot as plt


def run_camera_efficiency(
    config, output_directory, label=None, telescope_model=None, site_model=None
):
    """Run Shower, NSB and Muon camera-efficiency calculations.

    Parameters
    ----------
    config : dict
        Application configuration containing model, telescope and pointing settings.
    output_directory : pathlib.Path
        Directory for the combined YAML summary.
    label : str, optional
        Log label for this calculation.
    telescope_model, site_model : object, optional
        Initialized models to reuse; both are initialized when either is omitted.

    Returns
    -------
    dict
        Combined summary, summary path and the generated result and plot paths.
    """
    logger = logging.getLogger(__name__)
    if telescope_model is None or site_model is None:
        initialized_telescope, initialized_site, _ = initialize_simulation_models(
            label=label,
            model_version=config["model_version"],
            site=config["site"],
            telescope_name=config["telescope"],
        )
        telescope_model = telescope_model or initialized_telescope
        site_model = site_model or initialized_site

    results = {}
    wavelength_tables = []
    for efficiency_type in ("Shower", "NSB", "Muon"):
        efficiency = CameraEfficiency(
            label=label,
            config_data=config,
            efficiency_type=efficiency_type,
            telescope_model=telescope_model,
            site_model=site_model,
        )
        efficiency.simulate()
        efficiency.analyze(force=True)
        results |= efficiency.results_summary()
        figure = efficiency.plot_efficiency(save_fig=True)
        plt.close(figure)
        wavelength_tables.append(_result_path(efficiency))
        if efficiency.efficiency_type == "nsb":
            efficiency.dump_nsb_pixel_rate()
        if efficiency.efficiency_type == "muon":
            efficiency.calc_partial_efficiency()

    output_path = Path(output_directory)
    summary_path = output_path / names.generate_file_name(
        file_type="telescope_throughput_summary",
        suffix=".yml",
        site=config["site"],
        telescope_model_name=config["telescope"],
        zenith_angle=config["zenith_angle"].value,
        azimuth_angle=config["azimuth_angle"].value,
    )
    logger.info("Writing results summary to %s", summary_path)
    write_data_to_file(results, summary_path, unique_lines=True)
    plot_paths = sorted(
        path for path in output_path.iterdir() if path.suffix.lower() in {".png", ".pdf", ".svg"}
    )
    return {
        "results": results,
        "summary_path": summary_path,
        "artifacts": sorted(set(wavelength_tables + plot_paths)),
    }


def _result_path(efficiency):
    """Return the public result location using the calculator's file-name contract."""
    return efficiency.output_dir / names.generate_file_name(
        file_type="telescope_throughput",
        suffix=".ecsv",
        site=efficiency.telescope_model.site,
        telescope_model_name=efficiency.telescope_model.name,
        zenith_angle=efficiency.config["zenith_angle"],
        azimuth_angle=efficiency.config["azimuth_angle"],
        label=efficiency.efficiency_type,
    )
