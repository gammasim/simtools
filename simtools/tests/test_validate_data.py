#!/usr/bin/python3

import logging
import pytest

from astropy.table import Column
from astropy.table import Table
from astropy import units as u

import simtools.util.validate_data as ds

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def get_reference_columns():
    """
    return a test reference data column definition

    """
    return {
        'wavelength': {
            'description': 'wavelength',
            'required_column': True,
            'unit': 'nm',
            'type': 'float32',
            'required_range': {'unit': 'nm', 'min': 300, 'max': 700}},
        'qe': {
            'description': 'average quantum or photon detection efficiency',
            'required_column': True,
            'unit': 'dimensionless',
            'type': 'float32',
            'allowed_range': {'unit': 'unitless', 'min': 0.0, 'max': 1.0}
        },
        'abc': {
            'description': 'not required',
            'required_column': False,
            'unit': 'kg',
            'type': 'float32',
            'allowed_range': {'unit': 'kg', 'min': 0.0, 'max': 100.0}
        }
    }


def test__interval_check_allow_range():

    data_validator = ds.DataValidator(None, None)

    assert data_validator._interval_check((0.1, 0.9), (0., 1.), 'allowed_range') == True
    assert data_validator._interval_check((0., 1.), (0., 1.), 'allowed_range') == True

    assert data_validator._interval_check((-1., 0.9), (0., 1.), 'allowed_range') == False
    assert data_validator._interval_check((0., 1.1), (0., 1.), 'allowed_range') == False
    assert data_validator._interval_check((-1., 1.1), (0., 1.), 'allowed_range') == False


def test__interval_check_required_range():

    data_validator = ds.DataValidator(None, None)

    assert data_validator._interval_check((250., 700.), (300., 600), 'required_range') == True
    assert data_validator._interval_check((300., 600.), (300., 600), 'required_range') == True

    assert data_validator._interval_check((350., 700.), (300., 600), 'required_range') == False
    assert data_validator._interval_check((300., 500.), (300., 600), 'required_range') == False
    assert data_validator._interval_check((350., 500.), (300., 600), 'required_range') == False


def test__check_range():

    data_validator = ds.DataValidator(None, None)

    data_validator = ds.DataValidator(
        get_reference_columns(),
        None
    )

    col_1 = Column(name='qe', data=[0.1, 0.5], dtype='float32')
    data_validator._check_range(
        col_1.name, col_1.min(), col_1.max(),
        'allowed_range')
    col_w = Column(name='wavelength', data=[250., 750.], dtype='float32')
    data_validator._check_range(
        col_w.name, col_w.min(), col_w.max(),
        'required_range')

    col_2 = Column(name='key_error', data=[0.1, 0.5], dtype='float32')
    with pytest.raises(KeyError):
        data_validator._check_range(
            col_2.name, col_2.min(), col_2.max(),
            'allowed_range')

    with pytest.raises(KeyError):
        data_validator._check_range(
            col_w.name, col_w.min(), col_w.max(),
            'failed_range')


def test__column_units():

    data_validator = ds.DataValidator(
        get_reference_columns(),
        None
    )

    table_1 = Table()
    table_1['wavelength'] = Column([300.0, 350.0], unit='nm', dtype='float32')
    table_1['qe'] = Column([0.1, 0.5], dtype='float32')
    table_1['qe'] = Column([0.1, 0.5], unit=None, dtype='float32')
    table_1['qe'] = Column([0.1, 0.5], unit='dimensionless', dtype='float32')

    for col in table_1.itercols():
        data_validator._check_and_convert_units(col)

    table_2 = Table()
    table_2['wavelength'] = Column([300.0, 350.0], unit='nm', dtype='float32')
    table_2['wrong_column'] = Column([0.1, 0.5], dtype='float32')

    with pytest.raises(KeyError, match=r"'wrong_column'"):
        for col in table_2.itercols():
            data_validator._check_and_convert_units(col)

    table_3 = Table()
    table_3['wavelength'] = Column([300.0, 350.0], unit='kg', dtype='float32')

    with pytest.raises(u.core.UnitConversionError):
        for col in table_3.itercols():
            data_validator._check_and_convert_units(col)


def test__check_required_columns():

    data_validator = ds.DataValidator(
        get_reference_columns(),
        None
    )

    table_1 = Table()
    table_1['wavelength'] = Column([300.0, 350.0], unit='nm', dtype='float32')
    table_1['qe'] = Column([0.1, 0.5], dtype='float32')

    data_validator.data_table = table_1
    data_validator._check_required_columns()

    table_2 = Table()
    table_2['wavelength'] = Column([300.0, 350.0], unit='nm', dtype='float32')

    data_validator.data_table = table_2

    with pytest.raises(KeyError, match=r"'qe'"):
        data_validator._check_required_columns()
