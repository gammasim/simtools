#!/usr/bin/python3

import logging

from simtools.model.site_model import SiteModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_site_model(db_config):
    _south = SiteModel(site="South", mongo_db_config=db_config, label="testing-sitemodel")

    print(_south._parameters)
