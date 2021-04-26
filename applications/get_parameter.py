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
        '-s',
        '--site',
        help='Site (North or South)',
        type=str,
        required=True
    )
    parser.add_argument(
        '-t',
        '--telescope',
        help='Telescope type (e.g. LST-1, SST-D)',
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
        '--model_version',
        help=(
            'Parameter version. If no version is provided, '
            'the entries of the last 5 versions are printed.'
        ),
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

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    if not cfg.get('useMongoDB'):
        raise ValueError('This application works only with MongoDB and you asked not to use it')

    logger.info('TEST')

    db = db_handler.DatabaseHandler()

    if args.model_version == 'all':
        raise NotImplemented('Printing last 5 versions is not implemented yet.')
    else:
        version = args.model_version
    pars = db.getModelParameters(args.site, args.telescope, version)
    print()
    pprint(pars[args.parameter])
    print()
