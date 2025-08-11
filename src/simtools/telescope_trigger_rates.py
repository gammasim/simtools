"""Trigger rate calculation for telescopes and arrays of telescopes."""

import logging

from simtools.io import ascii_handler, io_handler
from simtools.layout.array_layout_utils import get_array_elements_from_db_for_layouts
from simtools.simtel.simtel_io_event_histograms import SimtelIOEventHistograms

_logger = logging.getLogger(__name__)


def telescope_trigger_rates(args_dict, db_config):
    """
    Calculate trigger rates for single telescopes or arrays of telescopes.

    Main function to read event data, fill histograms, and derive trigger rates.


    """
    if args_dict.get("array_layout_name"):
        telescope_configs = get_array_elements_from_db_for_layouts(
            args_dict["array_layout_name"],
            args_dict.get("site"),
            args_dict.get("model_version"),
            db_config,
        )
    else:
        telescope_configs = ascii_handler.collect_data_from_file(args_dict["telescope_ids"])[
            "telescope_configs"
        ]

    for array_name, telescope_ids in telescope_configs.items():
        _logger.info(
            f"Processing file: {args_dict['event_data_file']} with telescope config: {array_name}"
        )
        histograms = SimtelIOEventHistograms(
            args_dict["event_data_file"], array_name=array_name, telescope_list=telescope_ids
        )
        histograms.fill()
        if args_dict["plot_histograms"]:
            histograms.plot(
                output_path=io_handler.IOHandler().get_output_directory(),
            )
