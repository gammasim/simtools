#!/usr/bin/python3

"""
    Summary
    -------
    This application derives the parameter mirror_reflection_random_angle \
    (mirror roughness, also called rnda here) \
    for a given set of measured photon containment diameter (e.g. 80\% containment) \
    of individual mirrors.  The mean value of the measured containment diameters \
    in cm is required and its sigma can be given optionally but will only be used for plotting. \

    The individual mirror focal length can be taken into account if a mirror list which contains \
    this information is used from the :ref:`Model Parameters DB` or if a new mirror list is given \
    through the argument mirror_list. Random focal lengths can be used by turning on the argument \
    use_random_focal length and a new value for it can be given through the argument random_flen.

    The algorithm works as follow: A starting value of rnda is first defined as the one taken \
    from the :ref:`Model Parameters DB` \
    (or alternatively one may want to set it using the argument rnda).\
    Secondly, ray tracing simulations are performed for single mirror configurations for each \
    mirror given in the mirror_list. The mean simulated containment diameter for all the mirrors \
    is compared with the mean measured containment diameter. A new value of rnda is then defined \
    based on the sign of the difference between measured and simulated containment diameters and a \
    new set of simulations is performed. This process repeat until the sign of the \
    difference changes, meaning that the two final values of rnda brackets the optimal. These \
    two values are used to find the optimal one by a linear \
    interpolation. Finally, simulations are performed by using the the interpolated value \
    of rnda, which is defined as the desired optimal.

    A option no_tuning can be used if one only wants to simulate one value of rnda and compare \
    the results with the measured ones.

    The results of the tuning are plotted. See examples of the containment diameter \
    D80 vs rnda plot, on the left, and the D80 distributions, on the right. \

    .. _deriva_rnda_plot:
    .. image:: images/derive_mirror_rnda_North-MST-FlashCam-D.png
      :width: 49 %
    .. image:: images/derive_mirror_rnda_North-MST-FlashCam-D_D80-distributions.png
      :width: 49 %

    This application uses the following simulation software tools:
        - sim_telarray/bin/sim_telarray
        - sim_telarray/bin/rx (optional)

    Command line arguments
    ----------------------
    telescope (str, required)
        Telescope name (e.g. North-LST-1, South-SST-D, ...)
    model_version (str, optional)
        Model version (default=prod4)
    containment_mean (float, required)
        Mean of measured containment diameter [cm]
    containment_sigma (float, optional)
        Std dev of measured containment diameter [cm]
    containment_fraction (float, required)
        Containment fractio for diameter calculation (typically 0.8)
    rnda (float, optional)
        Starting value of mirror_reflection_random_angle. If not given, the value from the \
        default model will be used.
    2f_measurement (file, optional)
        File with results from 2f measurements including mirror panel raddii and spot size \
        measurements
    d80_list (file, optional)
        File with single column list of measured D80 [cm]. It is used only for plotting the D80 \
        distributions. If given, the measured distribution will be plotted on the top of the \
        simulated one.
    mirror_list (file, optional)
        Mirror list file (in sim_telarray format) to replace the default one. It should be used \
        if measured mirror focal lengths need to be taken into account.
    use_random_flen (activation mode, optional)
        Use random focal lengths, instead of the measured ones. The argument random_flen can be \
        used to replace the default random_focal_length from the model.
    random_flen (float, optional)
        Value to replace the default random_focal_length. Only used if use_random_flen \
        is activated.
    no_tuning (activation mode, optional)
        Turn off the tuning - A single case will be simulated and plotted.
    test (activation mode, optional)
        If activated, application will be faster by simulating only few mirrors.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    MST - Prod5 (07.2020)

    Runtime about 3 min.

    .. code-block:: console

        python applications/derive_mirror_rnda.py --site North --telescope MST-FlashCam-D \
            --containment_mean 1.4 --containment_sigma 0.16 --containment_fraction 0.8 \
            --mirror_list mirror_MST_focal_lengths.dat --d80_list mirror_MST_D80.dat \
            --rnda 0.0075


    Expected output:

    .. code-block:: console

        Measured Containment Diameter (80% containment):
        Mean = 1.400 cm, StdDev = 0.160 cm

        Simulated Containment Diameter (80% containment):
        Mean = 1.401 cm, StdDev = 0.200 cm

        mirror_random_reflection_angle
        Previous value = 0.007500
        New value = 0.006378


    .. todo::

        * Change default model to default (after this feature is implemented in db_handler)
        * Fix the setStyle. For some reason, sphinx cannot built docs with it on.
"""


import logging

import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.table import QTable

import simtools.config as cfg
import simtools.util.general as gen
from simtools.ray_tracing import RayTracing
from simtools.model.telescope_model import TelescopeModel
import simtools.util.commandline_parser as argparser
import simtools.util.workflow_description as workflow_config
import simtools.util.model_data_writer as writer


def parse():
    """
    Parse command line configuration

    """

    parser = argparser.CommandLineParser()
    parser.initialize_telescope_model_arguments()
    psf_group = parser.add_mutually_exclusive_group()
    psf_group.add_argument(
        "--psf_measurement_containment_mean",
        help="Mean of measured PSF containment diameter [cm]",
        type=float, required=False
    )
    psf_group.add_argument(
        "--psf_measurement",
        help="Results from PSF measurements for each mirror panel spot size",
        type=str, required=False,
    )
    parser.add_argument(
        "--psf_measurement_containment_sigma",
        help="Std dev of measured PSF containment diameter [cm]",
        type=float, required=False
    )
    parser.add_argument(
        "--containment_fraction",
        help="Containment fraction for diameter calculation (in interval 0,1)",
        type=argparser.CommandLineParser.efficiency_interval, required=False,
        default=0.8,
    )
    parser.add_argument(
        "--rnda",
        help="Starting value of mirror_reflection_random_angle",
        type=float, required=False,
        default=0.0,
    )
    parser.add_argument(
        "--mirror_list",
        help=("Mirror list file to replace the default one. It should be used if"
              " measured mirror focal lengths need to be accounted"),
        type=str, required=False,
    )
    parser.add_argument(
        "--use_random_flen",
        help=("Use random focal lengths. Read value for random_focal_length parameter read"
              " from DB or provide by using the argument random_flen."),
        action="store_true", required=False,
    )
    parser.add_argument(
        "--random_flen",
        help="Value to replace the default random_focal_length.",
        type=float, required=False,
    )
    parser.add_argument(
        "--no_tuning",
        help="no tuning of random_reflection_angle (a single case will be simulated).",
        action="store_true", required=False,
    )
    parser.initialize_toplevel_metadata_scheme()
    parser.initialize_default_arguments()
    return parser.parse_args()


def define_telescope_model(workflow):
    """
    Define telescope model and update configuration
    with mirror list and/or random focal length given
    as input

    Attributes
    ----------
    workflow_config: WorkflowDescription
       workflow configuration

    Returns
    ------
    tel TelescopeModel
        telescope model

    """

    tel = TelescopeModel(
        site=workflow.configuration('site'),
        telescopeModelName=workflow.configuration('telescope'),
        modelVersion=workflow.configuration('model_version'),
        label=workflow.label()
    )
    if workflow.configuration('mirror_list') is not None:
        mirrorListFile = cfg.findFile(name=workflow.configuration('mirror_list'))
        tel.changeParameter("mirror_list", workflow.configuration('mirror_list'))
        tel.addParameterFile("mirror_list", mirrorListFile)
    if workflow.configuration('random_flen') is not None:
        tel.changeParameter("random_focal_length", str(workflow.configuration('random_flen')))

    return tel


def print_and_write_results(workflow,
                            rndaStart, rndaOpt,
                            meanD80, sigD80,
                            resultsRnda, resultsMean, resultsSig):
    """
    Print results to screen write metadata and data files
    in the requested format

    """

    containment_fraction_percent = int(
        workflow.configuration('containment_fraction')*100)

    # Printing results to stdout
    print("\nMeasured D{:}:".format(containment_fraction_percent))
    if workflow.configuration('psf_measurement_containment_sigma') is not None:
        print(
            "Mean = {:.3f} cm, StdDev = {:.3f} cm".format(
                workflow.configuration('psf_measurement_containment_mean'),
                workflow.configuration('psf_measurement_containment_sigma'))
        )
    else:
        print("Mean = {:.3f} cm".format(
            workflow.configuration('psf_measurement_containment_mean')))
    print("\nSimulated D{:}:".format(containment_fraction_percent))
    print("Mean = {:.3f} cm, StdDev = {:.3f} cm".format(meanD80, sigD80))
    print("\nmirror_random_reflection_angle")
    print("Previous value = {:.6f}".format(rndaStart))
    print("New value = {:.6f}\n".format(rndaOpt))

    # Result table written to ecsv file using file_writer
    # First entry is always the best fit result
    result_table = QTable([
        [True]+[False]*len(resultsRnda),
        ([rndaOpt]+resultsRnda) * u.deg,
        ([0.]*(len(resultsRnda)+1)),
        ([0.]*(len(resultsRnda)+1)) * u.deg,
        ([meanD80]+resultsMean) * u.cm,
        ([sigD80]+resultsSig) * u.cm
        ], names=(
            'best_fit',
            'mirror_reflection_random_angle_sigma1',
            'mirror_reflection_random_angle_fraction2',
            'mirror_reflection_random_angle_sigma2',
            f'containment_radius_D{containment_fraction_percent}',
            f'containment_radius_sigma_D{containment_fraction_percent}'
        )
    )
    file_writer = writer.ModelDataWriter(workflow)
    file_writer.write_metadata()
    file_writer.write_data(result_table)


def get_psf_containment(logger, workflow):
    """
    Read measured single-mirror point-spread function (containment)
    from file and return mean and sigma

    """

    _psf_list = Table.read(
        workflow.configuration('psf_measurement'), format='ascii.ecsv')
    try:
        workflow.configuration(
             'psf_measurement_containment_mean',
             np.nanmean(np.array(_psf_list['psf_opt'].to('cm').value)))
        workflow.configuration(
             'psf_measurement_containment_sigma',
             np.nanstd(np.array(_psf_list['psf_opt'].to('cm').value)))
    except KeyError:
        logger.debug("Missing column for psf measurement (psf_opt) in {}".format(
            workflow.configuration('psf_measurement')))
        raise

    logger.info('Determined PSF containment to {:.4} +- {:.4} cm'.format(
         workflow.configuration('psf_measurement_containment_mean'),
         workflow.configuration('psf_measurement_containment_sigma')))


def main():

    args = parse()
    label = "derive_mirror_rnda"

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    workflow = workflow_config.WorkflowDescription(label=label, args=args)

    tel = define_telescope_model(workflow)

    if workflow.configuration('psf_measurement'):
        get_psf_containment(logger, workflow)

    def run(rnda):
        """Runs the simulations for one given value of rnda"""
        tel.changeParameter("mirror_reflection_random_angle", str(rnda))
        ray = RayTracing.fromKwargs(
            telescopeModel=tel,
            singleMirrorMode=True,
            mirrorNumbers=list(range(1, 10)) if workflow.configuration('test') else "all",
            useRandomFocalLength=workflow.configuration('use_random_flen'),
        )
        ray.simulate(test=False, force=True)  # force has to be True, always
        ray.analyze(force=True)

        return (
            ray.getMean("d80_cm").to(u.cm).value,
            ray.getStdDev("d80_cm").to(u.cm).value,
        )

    # First - rnda from previous model or from command line
    if workflow.configuration('rnda') != 0:
        rndaStart = workflow.configuration('rnda')
    else:
        rndaStart = tel.getParameter("mirror_reflection_random_angle")["Value"]
        if isinstance(rndaStart, str):
            rndaStart = rndaStart.split()
            rndaStart = float(rndaStart[0])

    logger.info("Start value for mirror_reflection_random_angle: {:}".format(
        rndaStart))

    resultsRnda = list()
    resultsMean = list()
    resultsSig = list()
    if workflow.configuration('no_tuning'):
        rndaOpt = rndaStart
    else:
        def collectResults(rnda, mean, sig):
            resultsRnda.append(rnda)
            resultsMean.append(mean)
            resultsSig.append(sig)

        stop = False
        meanD80, sigD80 = run(rndaStart)
        rnda = rndaStart
        signDelta = np.sign(
            meanD80 - workflow.configuration('psf_measurement_containment_mean'))
        collectResults(rnda, meanD80, sigD80)
        while not stop:
            newRnda = rnda - (0.1 * rndaStart * signDelta)
            meanD80, sigD80 = run(newRnda)
            newSignDelta = np.sign(
                meanD80 - workflow.configuration('psf_measurement_containment_mean'))
            stop = newSignDelta != signDelta
            signDelta = newSignDelta
            rnda = newRnda
            collectResults(rnda, meanD80, sigD80)

        # Linear interpolation using two last rnda values
        resultsRnda, resultsMean, resultsSig = gen.sortArrays(
            resultsRnda, resultsMean, resultsSig
        )
        rndaOpt = np.interp(
            x=workflow.configuration('psf_measurement_containment_mean'),
            xp=resultsMean, fp=resultsRnda)

    meanD80, sigD80 = run(rndaOpt)

    print_and_write_results(
        workflow,
        rndaStart, rndaOpt,
        meanD80, sigD80,
        resultsRnda, resultsMean, resultsSig)


if __name__ == "__main__":
    main()
