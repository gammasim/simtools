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
    - copy

TODO:
    
    - TelescopeModel: implement fromConfigFile method
    - SimtelRunner: files (log, photon, star etc) names
    - SimtelRunner: method to obtain script
    - SimtelRunner: force flag

GUIDELINES:

    - Any (functional) module must have a correspondent test module.
    These modules should be located at 'tests' and named as test_{module name}.py
    - Names should always be validated. (See util/names.py)
    - Generic utilities should go to util/
    - 

STYLE REMARKS:

    - Pep8, please
    - Breaking list of arguments when needed:
    function(
        par1,
        par2,
        par3
    )
    - 

DESIGN REMARKS:

    - Every functional class contains a 'label'.
    The label can be passed forward from lower level to higher level classes.
    In particular, label from TelescopeModel can be used in higher level classes.
    - Classes are not designed to be re-used, all parameters should be set when
    initializing and not changed afterwards. New parameters should mean new instance of the class.
    - filesLocation
    - A test flag (test=True/False) should exist always as possible. If True, it must provide a faster and simpler implementation.
    Spectially important for time comsuming simulations. The default must always be test=False.
    - A force flag (force=True/False) should exist in any case in which files are created. If False, the existing files
    should not be overwritten. The default must always be force=False.
