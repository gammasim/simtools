#!/usr/bin/python3

import logging

from ctamclib.telescope_model import TelescopeModel

logging.getLogger().setLevel(logging.DEBUG)

#if __name__ == '__main__':


def test():

    yamlDBPath = ('/home/prado/Work/Projects/CTA_MC/svn/Simulations/MCModelDescription/trunk/'
                  'configReports')

    tel = TelescopeModel(yamlDBPath=yamlDBPath, telescopeType='lst', site='south', version='prod4',
                         label='test-lst')

    print('TEST::Corrected telType: {}'.format(tel.telescopeType))
    print('TEST::Corrected site: {}'.format(tel.site))

    print(
        'TEST::Old mirror_reflection_random_angle:',
        tel.getParameter('mirror_reflection_random_angle')
    )
    print('TEST::Changing mirror_reflection_random_angle')
    tel.changeParameters(mirror_reflection_random_angle='0.0080 0 0')
    tel.addParameters(new_parameter='1')
    tel.removeParameters('new_parameter', 'mirror_reflection_random_angle')
    #print(
    #    'TEST::New mirror_reflection_random_angle:',
    #    tel.getParameter('mirror_reflection_random_angle')
    #)

    # Testing focal lenght type
    flen = tel.getParameter('focal_length')
    print('TEST::Focal Length', flen, type(flen))

    print('TEST::Exporting config file')
    # tel.exportConfigFile(loc='/home/prado/Work/Projects/CTA_MC/MCLib')
    tel.exportConfigFile()

    print('TEST::Config file: ', tel.getConfigFile())

    print('TEST::-----------------------------------')
    print('TEST:: Testing fromConfigFile')

    cfgFile = (
        '/afs/ifh.de/group/cta/scratch/prado/corsika_simtelarray/'
        'corsika6.9_simtelarray_18-11-07/sim_telarray/cfg/CTA/CTA-ULTRA6-SST-ASTRI.cfg'
    )
    tel = TelescopeModel.fromConfigFile(
        telescopeType='astri',
        site='south',
        # version='prod4',
        label='test-astri',
        configFileName=cfgFile
    )

    tel.exportConfigFile()

    print('TEST::---test_telescope_model done!---')
