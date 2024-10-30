#!/usr/bin/env python3
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


from pylab import *
sqr32 = sqrt(3) / 2.

results = loadtxt('bands6.out', unpack=True)

plot(results[0], sqrt(results[1]**2 + results[2]**2), 'k--')

try:
    reference = loadtxt('bands.dat', unpack=True)
except IOError:
    pass
else:
    plot(reference[0], reference[3], '.', color='0.7')

plot(results[0], results[3], '.', color='maroon')

xticks([0., pi*sqr32, pi*(sqr32+0.5), pi*(sqr32+1.5)], ['$\\Gamma$', 'M', 'K', '$\\Gamma$'])
xlim(0., pi*(sqr32+1.5))
grid(axis='x')

ylabel("$\\omega/c$")

tight_layout(pad=0.1)
show()
