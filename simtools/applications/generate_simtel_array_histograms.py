#!/usr/bin/python3

"""
    Summary
    -------
    This application allows to write sim_telarray histograms into pdf and hdf5 files.
    It accepts multiple lists of histograms files, a single list or a histogram file.
    Each histogram is plotted in a page of the pdf file if the --pdf option is activated.


    Command line arguments
    ----------------------
    hist_file_names (str, optional)
        Name of the histogram files to be plotted.
        It can be given as the histogram file names (more than one option allowed) or as a text
        file with the names of the histogram files in it.
    figure_name (str, required)
        File name for the pdf output (without extension).
    pdf (bool, optional)
        If set, histograms are saved into pdf files.
        One pdf file contains all the histograms found in the file.
        The name of the file is controlled via `output_file_name`.
    hdf5: bool
        If true, histograms are saved into hdf5 files.
        At least one of `pdf` and `hdf5` has to be activated.
    output_file_name (str, optional)
        The name of the output hdf5 (and/or pdf) files (without the path).
        If not given, `output_file_name` takes the name from the (first) input file
        (`hist_file_names`).
        If the output `output_file_name.hdf5` file already exists and `hdf5` is set, the tables
        associated to `hdf5` will be overwritten. The remaining tables, if any, will stay
        untouched.
    verbosity (str, optional)
        Log level to print.

    Raises
    ------
    TypeError:
        if argument passed through `hist_file_names` is not a file.

    Example
    -------
    .. code-block:: console

        simtools-generate-simtel-array-histograms --hist_file_names tests/resources/
            run2_gamma_za20deg_azm0deg-North-Prod5_test-production-5.hdata.zst
            --output_file_name test_hist_hdata --hdf5 --pdf

"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.simtel.simtel_histograms import SimtelHistograms


def _parse(label, description):
    """
    Parse command line configuration

    Parameters
    ----------
    label: str
        Label describing the application.
    description: str
        Description of the application.

    Returns
    -------
    CommandLineParser
        Command line parser object

    """
    config = configurator.Configurator(label=label, description=description)

    config.parser.add_argument(
        "--hist_file_names",
        help="Name of the histogram files to be plotted or the text file containing the list of "
        "histogram files.",
        nargs="+",
        required=True,
        type=str,
    )

    config.parser.add_argument(
        "--hdf5", help="Save histograms into a hdf5 file.", action="store_true", required=False
    )

    config.parser.add_argument(
        "--pdf", help="Save histograms into a pdf file.", action="store_true", required=False
    )

    config.parser.add_argument(
        "--output_file_name",
        help="Name of the hdf5 (and/or pdf) file where to save the histograms.",
        type=str,
        required=False,
        default=None,
    )

    config_parser, _ = config.initialize(db_config=False, paths=True)
    if not config_parser["pdf"]:
        if not config_parser["hdf5"]:
            config.parser.error("At least one argument is required: `--pdf` or `--hdf5`.")

    return config_parser


def main():
    label = Path(__file__).stem
    description = "Display the simtel_array histograms."
    io_handler_instance = io_handler.IOHandler()
    config_parser = _parse(label, description)
    output_path = io_handler_instance.get_output_directory(label, sub_dir="application-plots")
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(config_parser["log_level"]))
    logger.info("Starting the application.")

    # Building list of histograms from the input files
    histogram_files = []
    for one_file in config_parser["hist_file_names"]:
        try:
            if Path(one_file).suffix in [".zst", ".simtel", ".hdata"]:
                histogram_files.append(one_file)
            else:
                # Collecting hist files
                with open(one_file, encoding="utf-8") as file:
                    for line in file:
                        # Removing '\n' from filename, in case it is left there.
                        histogram_files.append(line.replace("\n", ""))
        except FileNotFoundError as exc:
            msg = f"{one_file} is not a file."
            logger.error(msg)
            raise FileNotFoundError from exc

    # If no output name is passed, the tool gets the name of the first histogram of the list
    if config_parser["output_file_name"] is None:
        config_parser["output_file_name"] = Path(histogram_files[0]).absolute().name
    output_file_name = Path(output_path).joinpath(f"{config_parser['output_file_name']}")

    # If the hdf5 output file already exists, it is overwritten
    if (Path(f"{config_parser['output_file_name']}.hdf5").exists()) and (config_parser["hdf5"]):
        msg = (
            f"Output hdf5 file {config_parser['output_file_name']}.hdf5 already exists. "
            f"Overwriting it."
        )
        logger.warning(msg)
        overwrite = True
    else:
        overwrite = False

    # Building SimtelHistograms
    simtel_histograms = SimtelHistograms(histogram_files)

    if config_parser["pdf"]:
        logger.debug(f"Creating the pdf file {output_file_name}.pdf")
        pdf_pages = PdfPages(f"{output_file_name}.pdf")

        if config_parser["test"]:
            number_of_histograms = 2
        else:
            number_of_histograms = simtel_histograms.number_of_histograms

        for i_hist in range(number_of_histograms):
            title = simtel_histograms.get_histogram_title(i_hist)

            logger.debug(f"Processing: {title}")

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            simtel_histograms.plot_one_histogram(i_hist, ax)

            plt.tight_layout()
            pdf_pages.savefig(fig)
            plt.clf()

        plt.close()
        pdf_pages.close()
        logger.info(f"Wrote histograms to the pdf file {output_file_name}.pdf")

    if config_parser["hdf5"]:
        logger.info(f"Wrote histograms to the hdf5 file {output_file_name}.hdf5")
        simtel_histograms.export_histograms(f"{output_file_name}.hdf5", overwrite=overwrite)


if __name__ == "__main__":
    main()
