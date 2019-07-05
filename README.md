Library to be used in the MC pipeline of CTA Observatory
Authors:
    
    - Raul R Prado (raul.prado@desy.de) 


REQUIREMENTS

    - logging
    - yaml
    - pathlib
    - os
    - numpy
    - astropy
    - subprocess
    - math
    - matplotlib

TODO:
    
    - TelescopeModel: implement fromConfigFile method
    - SimtelRunner: files (log, photon, star etc) names
    - SimtelRunner: method to obtain script
    - SimtelRunner: force flag

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
    - Classes are not designed to be re-used, all parameters should be set when
    initializing and not changed afterwards. New parameters should mean new object.
    - filesLocation
    - test flags
    - force flag
