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

import plask


class Generic2D(plask.Solver):
    def load_xpl(self, xpl, manager):
        for tag in xpl:
            plask.print_log('detail', tag.name, tag.attrs)


class Configured2D(plask.Solver):
    def load_xpl(self, xpl, manager):
        for tag in xpl:
            name = tag.name
            plask.print_log('detail', name, tag.attrs)
            for tag in tag:
                plask.print_log('detail', name, '>', tag.name, tag.attrs)
