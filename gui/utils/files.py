# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# coding: utf8

import sys
import os
from os.path import dirname as _d

__file__ = os.path.normpath(__file__)

def which(program):
    if os.path.split(program)[0]:
        if os.path.isfile(program) and os.access(program, os.X_OK):
            return program
    else:
        for path in [_d(sys.executable), os.path.join(_d(_d(_d(_d(_d(__file__))))), 'bin')]\
                + os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if os.path.isfile(exe_file) and os.access(exe_file, os.X_OK):
                return exe_file
    return None

def open_utf8(*args, **kwargs):
    try:
        return open(*args, encoding='utf-8', **kwargs)
    except TypeError:
        return open(*args, **kwargs)
