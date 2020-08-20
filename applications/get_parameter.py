#!/usr/bin/python3

import logging
import argparse
from pprint import pprint

from simtools import db_handler
import simtools.config as cfg
import simtools.util.general as gen

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Get a parameter entry from DB for a specific telescope. '
            'The application receives a parameter name and optionally a version. '
            'It then prints out the parameter entry. '
            'If no version is provided, the entries of the last 5 versions are printed.'
        )
    )
    parser.add_argument(
        '-t',
        '--tel_type',
        help='Telescope type (e.g. north-lst-1, south-sst-d)',
        type=str,
        required=True
    )
    parser.add_argument(
        '-p',
        '--parameter',
        help='Parameter name',
        type=str,
        required=True
    )
    parser.add_argument(
        '-v',
        '--version',
        help='Parameter version. If no version is provided, the entries of the last 5 versions are printed.',
        type=str,
        default='all'
    )
    parser.add_argument(
        '-V',
        '--verbosity',
        dest='logLevel',
        action='store',
        default='info',
        help='Log level to print (default is INFO)'
    )

    args = parser.parse_args()

    logger = logging.getLogger('get_parameter')
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    if not cfg.get('useMongoDB'):
        raise ValueError('This application works only with MongoDB and you asked not to use it')

    db = db_handler.DatabaseHandler(logger.name)

    if args.version == 'all':
        raise NotImplemented('Printing last 5 versions is not implemented yet.')
    else:
        version = args.version
    pars = db.getModelParameters(args.tel_type, version)
    print()
    pprint(pars[args.parameter])
    print()




    