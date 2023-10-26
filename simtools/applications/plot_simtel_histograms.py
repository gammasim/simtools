#!/usr/bin/python3

"""
    Summary
    -------
    This application plots and prints sim_telarray histograms into pdf file.
    It accepts multiple lists of histograms files that are plot together,
    side-by-side. Each histogram is plotted in a page of the pdf.


    Command line arguments
    ----------------------
    hist_file_names (str, optional)
        Name of the histogram files to be plotted.
        It can be given as the histogram file names (more than one option allowed) or as a text
            file with the names of the histogram files in it.
    figure_name (str, required)
        File name for the pdf output (without extension).
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    .. code-block:: console

        simtools-plot-simtel-histograms --hist_file_names
            ./tests/resources/run201_proton_za20deg_azm0deg_North_TestLayout_test-prod.simtel.zst
            --figure_name histograms

"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.simtel.simtel_histograms import SimtelHistograms


def main():
    config = configurator.Configurator(
        label=Path(__file__).stem,
        description=("Plots sim_telarray histograms."),
    )
    input_group = config.parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--hist_file_names",
        help="Name of the histogram files to be plotted or the text file containing the list of "
        "histogram files.",
        nargs="+",
        type=str,
    )
    config.parser.add_argument(
        "--figure_name", help="File name for the pdf output.", type=str, required=True
    )

    args_dict, _ = config.initialize()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    histogram_file_list = []
    if args_dict["file_lists"]:
        n_lists = len(args_dict["file_lists"])
        for one_list in args_dict["file_lists"]:
            # Collecting hist files
            histogram_files = []
            with open(one_list, encoding="utf-8") as file:
                for line in file:
                    # Removing '\n' from filename, in case it is left there.
                    histogram_files.append(line.replace("\n", ""))
            histogram_file_list.append(histogram_files)
    else:
        histogram_file_list = [args_dict["hist_file_names"]]
        n_lists = 1

    simtel_histograms = []
    for i_file in range(n_lists):
        # Building SimtelHistograms
        sh = SimtelHistograms(histogram_file_list[i_file])
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
