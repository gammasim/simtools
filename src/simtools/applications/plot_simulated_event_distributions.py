#!/usr/bin/python3

"""Plot simulated event distributions for shower and/or triggered event data."""

import glob
from pathlib import Path

import yaml

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.production_configuration.trigger_histograms import load_event_data_histograms
from simtools.visualization import plot_simtel_event_histograms

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "trigger_histogram_file",
        help="Trigger-histogram HDF5 file or threshold-scan glob pattern.",
        type=str,
        required=True,
    ),
    cli.ArgumentDefinition(
        "array_layout_name",
        help=(
            "Optional array layout name to select from a precomputed trigger-histogram "
            "file. If omitted, plot all layouts available in the file."
        ),
        type=str,
        default=None,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    needs=("corsika",),
)

_THRESHOLD_UNITS = {
    "discriminator_threshold": "mV",
    "asum_threshold": "mV",
    "dsum_threshold": "count",
}
_TRIGGERED_DISTRIBUTIONS = {
    "energy",
    "core_distance",
    "angular_distance",
    "core_distance_vs_energy",
    "angular_distance_vs_energy",
}


def _plot_threshold_scan(pattern, output_dir, array_name):
    """Plot triggered distributions for each threshold in histogram product manifests."""
    if not array_name:
        raise ValueError("array_layout_name is required for a threshold scan")
    path = Path(pattern)
    root = Path(path.anchor or ".")
    paths = sorted(root.glob(str(path.relative_to(root))))
    if not paths:
        raise FileNotFoundError(f"No trigger-histogram files match {pattern}")
    directories = set()
    for path in paths:
        metadata = yaml.safe_load(path.with_suffix(".yml").read_text(encoding="utf-8"))
        overrides = metadata["configuration"]["model_parameter_overrides"][array_name]
        selected = [
            (name, overrides[name]["value"]) for name in _THRESHOLD_UNITS if name in overrides
        ]
        if len(selected) != 1:
            raise ValueError(f"Expected one trigger threshold for {array_name} in {path}")
        parameter, value = selected[0]
        directory = Path(output_dir) / f"threshold-{value}{_THRESHOLD_UNITS[parameter]}"
        if directory in directories:
            raise ValueError(f"Duplicate threshold output directory: {directory}")
        directories.add(directory)
        loaded = load_event_data_histograms(path, array_names=[array_name])
        if len(loaded) != 1:
            raise ValueError(f"Expected one histogram for {array_name} in {path}")
        _, record = loaded[0]
        histograms = {
            name: record.histograms[name]
            for name in _TRIGGERED_DISTRIBUTIONS
            if name in record.histograms
        }
        if not histograms:
            raise ValueError(f"No triggered distributions for {array_name} in {path}")
        directory.mkdir(parents=True, exist_ok=True)
        plot_simtel_event_histograms.plot(histograms, output_path=directory)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    output_dir = app_context.paths.get_output_directory()

    app_context.logger.info(
        f"Loading trigger histogram file from: {app_context.args['trigger_histogram_file']}"
    )
    array_layout_name = app_context.args.get("array_layout_name")
    selected_names = [array_layout_name] if array_layout_name else None
    if glob.has_magic(app_context.args["trigger_histogram_file"]):
        _plot_threshold_scan(
            app_context.args["trigger_histogram_file"], output_dir, array_layout_name
        )
        return
    loaded_histograms = load_event_data_histograms(
        app_context.args["trigger_histogram_file"],
        array_names=selected_names,
    )
    if selected_names and not loaded_histograms:
        raise ValueError(
            f"Array layout '{array_layout_name}' not found in histogram file "
            f"'{app_context.args['trigger_histogram_file']}'."
        )
    plot_simtel_event_histograms.plot_trigger_histograms(loaded_histograms, output_dir)


if __name__ == "__main__":
    main()
