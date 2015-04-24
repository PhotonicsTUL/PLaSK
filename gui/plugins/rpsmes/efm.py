# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation; either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from pylab import pi

from gui.qt import QtGui, QtCore

import gui
import os


class Material(object):
    def __init__(self, name, nr, ar, ng, ag, tnr, tar, tng, tag, lam0):
        self.name = name
        self.nr = nr
        self.ng = ng
        self.ar = ar
        self.ng = ag
        self.tar = tar
        self.tag = tag
        self.tar = tar
        self.tag = tag
    def __eq__(self, other):
        return self.nr == other.nr and self.ng == other.ng and self.ar == other.ar and self.ag == other.ag and \
               self.tnr == other.tnr and self.tng == other.tng and self.tnr == other.tar and self.tng == other.tag

materials = []


class Layer(object):
    def __init__(self):
        pass


def read_efm(fname):

    ifile = open(fname)

    # Set-up generator, which skips empty lines, strips the '\n' character, and splits line by tabs
    def Input(ifile):
        for line in ifile:
            if line[-1] == "\n": line = line[:-1]
            if line.strip(): yield line.split()
    input = Input(ifile)

    def skip(n):
        for _ in range(n):
            input.next()

    # Read header
    skip(3)
    _, nl = input.next()
    skip(5)
    _, lam0 = input.next()
    lam0 *= 1e3
    skip(5)
    _, m = input.next()
    _, vre = input.next()
    _, vim = input.next()
    lam_start = lam0 / (1. - (vre+1j*vim)/2.)
    skip(3)

    for i in range(nl):
        names = input.next()
        lines = [input.next() for j in range(3)]

