#!/usr/bin/python3

'''
    Summary
    -------
    This application adds a file to the DB.

    The name and location of the file are required.
    This application should complement the ones for updating parameters \
    and adding entries to the DB.

    Command line arguments
    ----------------------
    fileName (str, required)
        Name of the file to upload including the full path.
        If no path is given, the file is assumed to be in the CWD.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    uploading a dummy file.

    .. code-block:: console

        python applications/add_file_to_db.py -f data/data-to-upload/test-data.dat
'''

import logging
import argparse

from simtools import db_handler
import simtools.util.general as gen


def userConfirm():
    '''
    Ask the user to enter Y or N (case-insensitive).

    Returns
    -------
    bool: True if the answer is Y/y.
    '''

    answer = ''
    while answer not in ['y', 'n']:
        answer = input('Is this OK? [Y/N]').lower()

    return answer == 'y'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Add a file to the DB.'
        )
    )
    parser.add_argument(
        '-f',
        '--fileName',
        default='./',
        help='The file name to upload. If no path is given, the file is assumed to be in the CWD.',
        type=str,
        required=True
    )
    parser.add_argument(
        '-v',
        '--verbosity',
        dest='logLevel',
        action='store',
        default='info',
        help='Log level to print (default is INFO)'
    )

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    db = db_handler.DatabaseHandler()

    print('Should I insert the file {} to the DB?'.format(args.fileName))
    if userConfirm():
        db.insertFileToDB(args.fileName)
        logger.info('File inserted to DB')
    else:
        logger.info('Aborted, did not insert file to the DB')
