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

    Raises
    ------
    TypeError:
        if argument passed through `hist_file_names` is not a file.

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

    config.parser.add_argument(
        "--hist_file_names",
        help="Name of the histogram files to be plotted or the text file containing the list of "
        "histogram files.",
        nargs="+",
        required=True,
        type=str,
    )
    config.parser.add_argument(
        "--figure_name", help="File name for the pdf output.", type=str, required=True
    )

    args_dict, _ = config.initialize()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))
    n_lists = len(args_dict["hist_file_names"])
    logger.info(f"Found {n_lists} hist_file_names. Opening them.")
    histogram_files = []
    for one_file in args_dict["hist_file_names"]:
        if Path(one_file).is_file():
            if Path(one_file).suffix in [".zst", ".simtel"]:
                histogram_files.append(one_file)
            else:
                # Collecting hist files
                with open(one_file, encoding="utf-8") as file:
                    for line in file:
                        # Removing '\n' from filename, in case it is left there.
                        histogram_files.append(line.replace("\n", ""))
        else:
            msg = f"{one_file} is not a file."
            logger.error(msg)
            raise TypeError

    # Building SimtelHistograms
    logger.info(f"Histograms will be produced for {histogram_files}.")
    simtel_histograms = SimtelHistograms(histogram_files)

    # Checking if it is needed to add the pdf extension to the file name
    if Path(args_dict["figure_name"]).suffix == "pdf":
        fig_name = args_dict["figure_name"]
    else:
        fig_name = args_dict["figure_name"] + ".pdf"
    logger.info(f"Creating pdf file here '{fig_name}'.")
    pdf_pages = PdfPages(fig_name)
    for i_hist in range(n_lists * simtel_histograms.number_of_histograms):
        title = simtel_histograms.get_histogram_title(i_hist)

        logger.debug(f"Processing: {title}")

        fig, axs = plt.subplots(1, n_lists, figsize=(6 * n_lists, 6))
        simtel_histograms.plot_one_histogram(i_hist, axs)

        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.clf()

    plt.close()
    pdf_pages.close()
    logger.info(f"Finalized writing histograms to '{fig_name}'.")


if __name__ == "__main__":
    main()
