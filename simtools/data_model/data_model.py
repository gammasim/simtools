class InvalidSchemaList(Exception):
    """
    Exception raised for requests of unknown schema lists
    """


def top_level_reference_schema():
    """
    Reference schema following the CTA Top-Level Data Model.

    This metadata schema is used for gammasim-tools data products.

    Returns
    -------
    dict with reference schema


    """

    return {
        "CTA": {
            "REFERENCE": {"VERSION": "1.0.0"},
            "CONTACT": {"ORGANIZATION": None, "NAME": None, "EMAIL": None},
            "PRODUCT": {
                "DESCRIPTION": None,
                "CREATION_TIME": None,
                "ID": None,
                "DATA": {
                    "CATEGORY": "SIM",
                    "LEVEL": "R0",
                    "TYPE": "service",
                    "MODEL": {"NAME": "simpipe-table", "VERSION": "0.1.0", "URL": None},
                },
                "FORMAT": None,
                "FILENAME": None,
                "VALID": {"START": None, "END": None},
            },
            "INSTRUMENT": {"SITE": None, "CLASS": None, "TYPE": None, "SUBTYPE": None, "ID": None},
            "PROCESS": {"TYPE": None, "SUBTYPE": None, "ID": None},
            "ACTIVITY": {
                "NAME": None,
                "TYPE": "software",
                "ID": None,
                "START": None,
                "END": None,
                "SOFTWARE": {"NAME": "gammasim-tools", "VERSION": None},
            },
            "CONTEXT": {
                "SIM": {
                    "ASSOCIATION": [
                        {"SITE": None, "CLASS": None, "TYPE": None, "SUBTYPE": None, "ID": None}
                    ],
                    "DOCUMENT": [
                        {
                            "TYPE": None,
                            "ID": None,
                            "LINK": None,
                        },
                    ],
                },
            },
        }
    }


def metadata_input_reference_schema():
    """
    Reference data model scheme for input metadata.
    Describes metadata provided for input to gammasim-tools applications.

    Returns
    -------
    dict with input reference schema

    """

    return {
        "REFERENCE": {"VERSION": {"type": "str", "required": True}},
        "CONTACT": {
            "ORGANIZATION": {"type": "str", "required": False},
            "NAME": {"type": "str", "required": True},
            "EMAIL": {"type": "email", "required": True},
        },
        "INSTRUMENT": metadata_input_reference_document_list("instrumentlist"),
        "PROCESS": {
            "TYPE": {"type": "str", "required": True},
            "ID": {"type": "str", "required": True},
        },
        "PRODUCT": {
            "DESCRIPTION": {"type": "str", "required": True},
            "CREATION_TIME": {"type": "datetime", "required": True},
            "FORMAT": {"type": "str", "required": False, "default": "ecsv"},
            "VALID": {
                "START": {"type": "datetime", "required": False, "default": "None"},
                "END": {"type": "datetime", "required": False, "default": "None"},
            },
            "ASSOCIATION": {"type": "instrumentlist", "required": True},
        },
        "CONTEXT": {
            "DOCUMENT": {"type": "documentlist", "required": False, "default": "None"},
        },
    }


def metadata_input_reference_document_list(schema_list):
    """
    Reference model data for input metata data of type DOCUMENT.

    Returns
    -------
    dict with input reference schema for DOCUMENT

    """
    if schema_list.lower() == "instrumentlist":
        return {
            "SITE": {"type": "str", "required": True},
            "CLASS": {"type": "str", "required": True},
            "TYPE": {"type": "str", "required": True},
            "SUBTYPE": {"type": "str", "required": False},
            "ID": {"type": "str", "required": True},
        }
    if schema_list.lower() == "documentlist":
        return {
            "TYPE": {"type": "str", "required": False},
            "ID": {"type": "str", "required": False},
            "LINK": {"type": "str", "required": False},
        }

    msg = "Invalid schema list: {}".format(schema_list)
    raise InvalidSchemaList(msg)


def workflow_configuration_schema():
    """
    Reference schmema for gammasim-tools workflow configuration.

    Returns
    -------
    dict with workflow configuration

    """

    return {
        "reference": {"version": "0.1.0"},
        "activity": {
            "name": None,
            "id": None,
            "description": None,
        },
        "datamodel": {
            "inputschema": None,
        },
        "product": {
            "description": None,
            "format": None,
            "filename": None,
        },
        "configuration": {
            "log_level": "INFO",
            "test": False,
        },
    }
