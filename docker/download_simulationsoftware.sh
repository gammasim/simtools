#!/bin/sh
#
# download CORSIKA / sim_telarray package from
# MPIK page
#
# requires CTA user name / password
#

VERSION="7.7"

curl -u CTA -O https://www.mpi-hd.mpg.de/hfm/CTA/MC/Software/Testing/corsika${VERSION}_simtelarray.tar.gz
