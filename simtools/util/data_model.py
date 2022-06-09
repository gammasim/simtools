def top_level_reference_schema():
    """
    Return datamodel reference schema
    derived from CTA Top-Level Data Model

    Returns
    -------
    dict with reference schema


    """

    return {
        'CTA': {
            'REFERENCE': {
                'VERSION': '1.0.0'
            },
            'PRODUCT': {
                'DESCRIPTION': None,
                'CREATION_TIME': None,
                'ID': None,
                'DATA': {
                    'CATEGORY': 'SIM',
                    'LEVEL': 'R0',
                    'TYPE': 'service',
                    'MODEL': {
                        'NAME': 'simpipe-table',
                        'VERSION': '0.1.0',
                        'URL': None
                    }
                },
                'FORMAT': None,
                'VALID': {
                    'START': None,
                    'END': None
                }
            },
            'INSTRUMENT': {
                'SITE': None,
                'CLASS': None,
                'TYPE': None,
                'SUBTYPE': None,
                'ID': None},
            'PROCESS': {
                'TYPE': None,
                'SUBTYPE': None,
                'ID': None
            },
            'CONTACT': {
                'ORGANIZATION': None,
                'NAME': None,
                'EMAIL': None
            },
            'ACTIVITY': {
                'NAME': None,
                'TYPE': 'software',
                'ID': None,
                'START': None,
                'END': None,
                'SOFTWARE': {
                    'NAME': 'gammasim-tools',
                    'VERSION': None
                }
            },
            'CONTEXT': {
                'SIM': {
                    'ASSOCIATION':
                    [
                        {
                            'SITE': None,
                            'CLASS': None,
                            'TYPE': None,
                            'SUBTYPE': None,
                            'ID': None
                        }
                    ],
                    'DOCUMENT':
                    [
                        {
                            'TYPE': None,
                            'ID': None,
                            'LINK': None,
                        },
                    ],
                },
            },
        }
    }


def userinput_reference_schema():
    """
    Return datamodel reference schema
    for user input

    Returns
    -------
    dict with user input reference schema


    """

    return {
        "REFERENCE": {
            "VERSION": {
                "type": "str",
                "required": True
            }
        },
        "CONTACT": {
            "ORGANIZATION": {
                "type": "str",
                "required": False
            },
            "NAME": {
                "type": "str",
                "required": True
            },
            "EMAIL": {
                "type": "email",
                "required": True
            }
        },
        "INSTRUMENT": {
            "SITE": {
                "type": "str",
                "required": True
            },
            "CLASS": {
                "type": "str",
                "required": True
            },
            "TYPE": {
                "type": "str",
                "required": True
            },
            "SUBTYPE": {
                "type": "str",
                "required": False
            },
            "ID": {
                "type": "str",
                "required": True
            }
        },
        "PROCESS": {
            "TYPE": {
                "type": "str",
                "required": True
            },
            "ID": {
                "type": "str",
                "required": True
            }
        },
        "PRODUCT": {
            "DESCRIPTION": {
                "type": "str",
                "required": True
            },
            "CREATION_TIME": {
                "type": "datetime",
                "required": True
            },
            "FORMAT": {
                "type": "str",
                "required": False,
                "default": "ecsv"
            },
            "DATA": {
                "type": "str",
                "required": True
            },
            "VALID": {
                "START": {
                    "type": "datetime",
                    "required": False,
                    "default": "None"
                },
                "END": {
                    "type": "datetime",
                    "required": False,
                    "default": "None"
                },
            },
            "DOCUMENT": {
                "type": "list",
                "required": False,
                "default": "None"
            },
            "ASSOCIATION": {
                "type": "instrumentlist",
                "required": True
            }
        }
    }
