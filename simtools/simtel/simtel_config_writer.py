#!/usr/bin/python3

import logging


__all__ = ['SimtelConfigWriter']


class SimtelConfigWriter:
    '''
    SimtelConfigWriter writes sim_telarray configuration files. \
    It is designed to be used by model classes (TelescopeModel and ArrayModel) only.

    Methods
    -------
    writeTelescopeConfigFile(configFilePath, parameters)
        Writes the sim_telarray config file for a single telescope.
    writeArrayConfigFile(configFilePath, layout, telescopeModel, siteParameters)
        Writes the sim_telarray config file for an array of telescopes.
    writeSingleMirrorListFile(
        mirrorNumber,
        mirrors,
        singleMirrorListFile,
        setFocalLengthToZero=False
    )
        Writes the sim_telarray mirror list file for a single mirror.
    '''
    TAB = ' ' * 3
    SITE_PARS = [
        'altitude',
        'atmospheric_transmission',
        'ref_lat',
        'ref_long',
        'array_coordinates',
        'atmospheric_profile',
        'magnetic_field'
    ]
    PARS_NOT_TO_WRITE = [
        'pixel_shape',
        'pixel_diameter',
        'lightguide_efficiency_angle_file',
        'lightguide_efficiency_wavelength_file',
        'ref_lat',
        'ref_long',
        'array_coordinates',
        'atmospheric_profile',
        'magnetic_field'
    ]
    COMMON_PARS = {
        'trigger_telescopes': 1,
        'array_trigger': 'none',
        'trigger_telescopes': 2,
        'only_triggered_telescopes': 1,
        'array_window': 1000,
        'output_format': 1,
        'mirror_list': 'none',
        'telescope_random_angle': 0.,
        'telescope_random_error': 0.,
        'convergent_depth': 0,
        'iobuf_maximum': 1000000000,
        'iobuf_output_maximum': 400000000,
        'multiplicity_offset': -0.5,
        'discriminator_pulse_shape': 'none',
        'discriminator_amplitude': 0.,
        'discriminator_threshold': 99999.,
        'fadc_noise': 0.,
        'asum_threshold': 0.,
        'asum_shaping_file': 'none',
        'asum_offset': 0.0,
        'dsum_threshold': 0,
        'fadc_pulse_shape': 'none',
        'fadc_amplitude': 0.,
        'fadc_pedestal': 100.,
        'fadc_max_signal': 4095,
        'fadc_max_sum': 16777215,
        'store_photoelectrons': 30,
        'pulse_analysis': -30,
        'sum_before_peak': 3,
        'sum_after_peak': 4
    }

    def __init__(self, site, modelVersion, layoutName=None, telescopeModelName=None, label=None):
        '''
        SimtelConfigWriter.

        Parameters
        ----------
        site: str
            South or North.
        modelVersion: str, required.
            Version of the model (ex. prod4).
        telescopeModelName: str, optional.
            Telescope model name.
        layoutName: str, optional.
            Layout name.
        label: str, optional
            Instance label. Important for output file naming.
        '''
        self._logger = logging.getLogger(__name__)
        self._logger.debug('Init SimtelConfigWriter')

        self._site = site
        self._modelVersion = modelVersion
        self._label = label
        self._layoutName = layoutName
        self._telescopeModelName = telescopeModelName

    def writeTelescopeConfigFile(self, configFilePath, parameters):
        '''
        Writes the sim_telarray config file for a single telescope.

        Parameters
        ----------
        configFilePath: str or Path
            Path of the file to write on.
        parameters: dict
            Model parameters in the same structure as used by the TelescopeModel class.
        '''
        with open(configFilePath, 'w') as file:
            self._writeHeader(file, 'TELESCOPE CONFIGURATION FILE')

            file.write('#ifdef TELESCOPE\n')
            file.write(
                '   echo Configuration for {}'.format(self._telescopeModelName)
                + ' - TELESCOPE $(TELESCOPE)\n'
            )
            file.write('#endif\n\n')

            for par in parameters.keys():
                if par in self.PARS_NOT_TO_WRITE:
                    continue
                value = parameters[par]['Value']
                file.write('{} = {}\n'.format(par, value))

    def writeArrayConfigFile(
        self,
        configFilePath,
        layout,
        telescopeModel,
        siteParameters
    ):
        '''
        Writes the sim_telarray config file for an array of telescopes.

        Parameters
        ----------
        configFilePath: str or Path
            Path of the file to write on.
        layout: LayoutArray
            LayoutArray object referent to the array model.
        telescopeModel: list of TelescopeModel
            List of TelescopeModel's as used by the ArrayModel.
        siteParameters: dict
            Site parameters.
        '''
        with open(configFilePath, 'w') as file:
            self._writeHeader(file, 'ARRAY CONFIGURATION FILE')

            # Be carefull with the formating - simtel is sensitive
            file.write('#ifndef TELESCOPE\n')
            file.write('# define TELESCOPE 0\n')
            file.write('#endif\n\n')

            # TELESCOPE 0 - global parameters
            file.write('#if TELESCOPE == 0\n')
            file.write(self.TAB + 'echo *****************************\n')
            file.write(self.TAB + 'echo Site: {}\n'.format(self._site))
            file.write(self.TAB + 'echo LayoutName: {}\n'.format(self._layoutName))
            file.write(self.TAB + 'echo ModelVersion: {}\n'.format(self._modelVersion))
            file.write(self.TAB + 'echo *****************************\n\n')

            # Writing site parameters
            self._writeSiteParameters(file, siteParameters)

            # Writing common parameters
            self._writeCommonParameters(file)

            # Maximum telescopes
            file.write(self.TAB + 'maximum_telescopes = {}\n\n'.format(len(telescopeModel)))

            # Default telescope - 0th tel in telescope list
            telConfigFile = (telescopeModel[0].getConfigFile(noExport=True).name)
            file.write('# include <{}>\n\n'.format(telConfigFile))

            # Looping over telescopes - from 1 to ...
            for count, telModel in enumerate(telescopeModel):
                telConfigFile = telModel.getConfigFile(noExport=True).name
                file.write('%{}\n'.format(layout[count].name))
                file.write('#elif TELESCOPE == {}\n\n'.format(count + 1))
                file.write('# include <{}>\n\n'.format(telConfigFile))
            file.write('#endif \n\n')
    # END writeSimtelArrayConfigFile

    def writeSingleMirrorListFile(
        self,
        mirrorNumber,
        mirrors,
        singleMirrorListFile,
        setFocalLengthToZero=False
    ):
        '''
        Writes the sim_telarray mirror list file for a single mirror.

        Parameters
        ----------
        mirrorNumber: int
            Mirror number.
        mirrors: Mirrors
            Mirrors object.
        singleMirrorListFile: str or Path
            Path of the file to write on.
        setFocalLengthToZero: bool
            Flag to set the focal length to zero.
        '''
        __, __, diameter, flen, shape = mirrors.getSingleMirrorParameters(mirrorNumber)

        with open(singleMirrorListFile, 'w') as file:
            self._writeHeader(file, 'MIRROR LIST FILE', '#')

            file.write('# Column 1: X pos. [cm] (North/Down)\n')
            file.write('# Column 2: Y pos. [cm] (West/Right from camera)\n')
            file.write('# Column 3: flat-to-flat diameter [cm]\n')
            file.write(
                '# Column 4: focal length [cm], typically zero = adapting in sim_telarray.\n'
            )
            file.write(
                '# Column 5: shape type: 0=circular, 1=hex. with flat side parallel to y, '
                '2=square, 3=other hex. (default: 0)\n'
            )
            file.write(
                '# Column 6: Z pos (height above dish backplane) [cm], typ. omitted (or zero)'
                ' to adapt to dish shape settings.\n'
            )
            file.write('#\n')
            file.write('0. 0. {} {} {} 0.\n'.format(
                diameter,
                flen if not setFocalLengthToZero else 0,
                shape
            ))
    # End of writeSingleMirrorListFile

    def _writeHeader(self, file, title, commentChar='%'):
        '''
        Writes a generic header. commenChar is the character to be used for comments, \
        which is differs among ctypes of config files.
        '''
        header = '{}{}\n'.format(commentChar, 50 * '=')
        header += '{} {}\n'.format(commentChar, title)
        header += '{} Site: {}\n'.format(commentChar, self._site)
        header += '{} ModelVersion: {}\n'.format(commentChar, self._modelVersion)
        header += (
            '{} TelescopeModelName: {}\n'.format(commentChar, self._telescopeModelName)
            if self._telescopeModelName is not None else ''
        )
        header += (
            '{} LayoutName: {}\n'.format(commentChar, self._layoutName)
            if self._layoutName is not None else ''
        )
        header += (
            '{} Label: {}\n'.format(commentChar, self._label)
            if self._label is not None else ''
        )
        header += '{}{}\n'.format(commentChar, 50 * '=')
        header += '{}\n'.format(commentChar)
        file.write(header)

    def _writeSiteParameters(self, file, siteParameters):
        ''' Writes site parameters. '''
        file.write(self.TAB + '% Site parameters\n')
        for par in siteParameters:
            if par in self.PARS_NOT_TO_WRITE:
                continue
            value = siteParameters[par]['Value']
            file.write(self.TAB + '{} = {}\n'.format(par, value))
        file.write('\n')

    def _writeCommonParameters(self, file):
        # Common parameters taken from CTA-PROD4-common.cfg
        # TODO: Store these somewhere else
        self._logger.warning('Common parameters are hardcoded!')
        for par, value in self.COMMON_PARS.items():
            file.write('   {} = {}\n'.format(par, value))
    # End of writeCommonParameters
