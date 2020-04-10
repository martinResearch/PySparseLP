# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------
# Copyright © 2016 Martin de la Gorce <martin[dot]delagorce[hat]gmail[dot]com>

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# -----------------------------------------------------------------------
"""Module to download netlib problems."""


import gzip
import os
import urllib.request

from .MPSparser import mps_parser


def get_problem(pbname):
    thisfilepath = os.path.dirname(os.path.abspath(__file__))

    filenameLP = os.path.join(thisfilepath, "data", "netlib", pbname.upper() + ".SIF")
    filenameSol = os.path.join(thisfilepath, "data", "perPlex", pbname.lower() + ".txt")

    if not os.path.isfile(filenameLP):
        urllib.request.urlretrieve(
            "ftp://ftp.numerical.rl.ac.uk/pub/cuter/netlib/%s.SIF" % pbname.upper(),
            filenameLP,
        )
    if not os.path.isfile(filenameSol):
        urllib.request.urlretrieve(
            "http://www.zib.de/koch/perplex/data/netlib/txt/%s.txt.gz" % pbname.lower(),
            filenameSol + ".gz",
        )
        fgz = gzip.open(filenameSol + ".gz")
        f = open(filenameSol, "w")
        for l in fgz.readlines():
            f.write(l)
        f.close()

    # netlib problems ftp://ftp.numerical.rl.ac.uk/pub/cuter/netlib.tar.gz
    # netlib exact solutions http://www.zib.de/koch/perplex/data/netlib/txt/

    fLP = open(filenameLP, "r")
    if filenameSol is not None:
        fSol = open(filenameSol, "r")
    else:
        fSol = None

    LPDict = mps_parser(fLP, fSol)
    return LPDict


if __name__ == "__main__":

    filenameLP = "./data/netlib/AFIRO.SIF"
    filenameSol = "./data/perPlex/afiro.txt"
    fLP = open(filenameLP, "r")
    fsol = open(filenameSol, "r")
    LP = mps_parser(fLP, fsol)
