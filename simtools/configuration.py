import logging


class Configuration:
    """
    Configuration handling application configuration.

    Allow to set configuration parameters by
    - command line arguments
    - configuration file (yml file)
    - environmental variables
    - configuration dict when calling the class

    """

    def __init__(self, config=None):
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init Configuration")

        if config is None:
            config = {}

    def __new__(cls):
        """
        Singleton definition

        """
        if not hasattr(cls, "instance"):
            cls.instance = super(Configuration, cls).__new__(cls)
        return cls.instance
