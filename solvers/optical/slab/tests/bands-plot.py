#!/usr/bin/python

from pylab import *
sqr32 = sqrt(3) / 2.

results = loadtxt('bands.out', unpack=True)

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

tight_layout(0.1)
show()
