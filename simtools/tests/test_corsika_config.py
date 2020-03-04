#!/usr/bin/python3

import logging

from simtools.telescope_model import TelescopeModel

logging.getLogger().setLevel(logging.DEBUG)

if __name__ == '__main__':

    yamlDBPath = (
        '/home/prado/Work/Projects/CTA_MC/svn/Simulations/SimulationModel/'
        'ModelDescription/trunk/configReports'
    )

    tel = TelescopeModel(yamlDBPath=yamlDBPath,
                         telescopeType='lst',
                         site='south',
                         version='prod4',
                         label='test-lst')

    print('TEST::Corrected telType: {}'.format(tel.telescopeType))
    print('TEST::Corrected site: {}'.format(tel.site))

    print('TEST::Old mirror_reflection_random_angle:',
          tel.getParameter('mirror_reflection_random_angle'))
    print('TEST::Changing mirror_reflection_random_angle')
    tel.changeParameters(mirror_reflection_random_angle='0.0080 0 0')
    tel.addParameters(new_parameter='1')
    tel.removeParameters('new_parameter', 'mirreflection_random_angle')
    print('TEST::New mirror_reflection_random_angle:', tel.getParameter('mirror_reflection_random_angle'))

    print('TEST::Exporting config file')
    # tel.exportConfigFile(loc='/home/prado/Work/Projects/CTA_MC/MCLib')
    tel.exportConfigFile()

    print('TEST::Config file: ', tel.getConfigFile())

    print('---test_telescope_model done!---')
