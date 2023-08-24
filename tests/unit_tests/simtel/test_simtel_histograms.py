#!/usr/bin/python3

import logging

from simtools.simtel.simtel_histograms import SimtelHistograms

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_histograms(io_handler):
    histogram_files = list()
    histogram_files.append(
        io_handler.get_input_data_file(
            file_name="run1_gamma_za20deg_azm0deg-North-Prod5_test-production-5.hdata.zst",
            test=True,
        )
    )
    histogram_files.append(
        io_handler.get_input_data_file(
            file_name="run2_gamma_za20deg_azm0deg-North-Prod5_test-production-5.hdata.zst",
            test=True,
        )
    )

    hists = SimtelHistograms(histogram_files=histogram_files, test=True)

    fig_name = io_handler.get_output_file(
        file_name="simtel_histograms.pdf", sub_dir="plots", dir_type="test"
    )
    hists.plot_and_save_figures(fig_name=fig_name)

    assert fig_name.exists()
