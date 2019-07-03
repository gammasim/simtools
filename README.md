
requirements
- logging
- yaml
- pathlib
- os

TODO:
    - TelescopeModel: implement fromConfigFile method
    - SimtelRunner: files (log, photon, star etc) names
    - SimtelRunner: method to obtain script
    - make loc a config parameter

STYLE REMARKS:

    - Breaking list of arguments when needed:
    function(
        par1,
        par2,
        par3
    )

    -

DESIGN REMARKS:

    - Every functional class contains "label".
    The label can be passes forward from more fundamental classes.
    In particular, label from TelescopeModel can be used in higher level classes.

    - filesLocation

    - test flags

    - force flag

    