#!/usr/bin/python3

"""
    Summary
    -------
    This application plots and prints sim_telarray histograms into pdf file.
    It accepts multiple lists of histograms files that are plot together,
    side-by-side. Each histogram is plotted in a page of the pdf.


    Command line arguments
    ----------------------
    file_lists (str, required)
        Text file containing the list of sim_telarray histogram files to be plotted. \
        Multiple text files can be given.
    figure_name (str, required)
        File name for the pdf output (without extension).
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    .. code-block:: console

        python applications/plot_simtel_histograms.py \
            --file_lists list_test1.txt list_test2.txt --figure_name histograms_comparison

"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import simtools.util.general as gen
from simtools.configuration import configurator
from simtools.simtel.simtel_histograms import SimtelHistograms


def main():

    config = configurator.Configurator(
        label=Path(__file__).stem,
        description=("Plots sim_telarray histograms."),
    )
    config.parser.add_argument(
        "--file_lists",
        help="File containing the list of histogram files to be plotted.",
        nargs="+",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--figure_name", help="File name for the pdf output.", type=str, required=True
    )

    args_dict, _ = config.initialize()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    n_lists = len(args_dict["file_lists"])
    simtel_histograms = list()
    for this_list_of_files in args_dict["file_lists"]:
        # Collecting hist files
        histogram_files = list()
        with open(this_list_of_files) as file:
            for line in file:
                # Removing '\n' from filename, in case it is left there.
                histogram_files.append(line.replace("\n", ""))

        # Building SimtelHistograms
        sh = SimtelHistograms(histogram_files)
        simtel_histograms.append(sh)

    # Checking if number of histograms is consistent
    number_of_hists = [sh.number_of_histograms for sh in simtel_histograms]
    # Converting list to set will remove the duplicated entries.
    # If all entries in the list are the same, len(set) will be 1
    if len(set(number_of_hists)) > 1:
        msg = (
            "Number of histograms in different sets of simulations is inconsistent"
            " - please make sure the simulations sets are consistent"
        )
        logger.error(msg)
        raise IOError(msg)

    # Plotting

    # Checking if it is needed to add the pdf extension to the file name
    if args_dict["figure_name"].split(".")[-1] == "pdf":
        fig_name = args_dict["figure_name"]
    else:
        fig_name = args_dict["figure_name"] + ".pdf"

    pdf_pages = PdfPages(fig_name)
    for i_hist in range(number_of_hists[0]):

        title = simtel_histograms[0].get_histogram_title(i_hist)

        logger.debug(f"Processing: {title}")

        fig, axs = plt.subplots(1, n_lists, figsize=(6 * n_lists, 6))

        if n_lists == 1:
            # If only one simulation set, axs is a single axes (not iterable)
            sh.plot_one_histogram(i_hist, axs)
        else:
            for sh, ax in zip(simtel_histograms, axs):
                sh.plot_one_histogram(i_hist, ax)

        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.clf()

    plt.close()
    pdf_pages.close()


if __name__ == "__main__":
    main()
