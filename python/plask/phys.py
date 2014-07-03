#coding: UTF-8
'''
Basic physical constants.

The following constants are defined here:

=============== ============================================================
:attr:`qe`      Elementary charge [C]
:attr:`c`       Speed of light [m/s]
:attr:`mu0`     Vacuum permeability [V·s/(A·m)]
:attr:`eps0`    Vacuum permittivity [pF/µm]
:attr:`h.J`     Planck constant [J·s]
:attr:`h.eV`    Planck constant [eV·s]
:attr:`SB`      Stefan-Boltzmann constant [W/(m\ :sup:`2`\ ·K\ :sup:`4`\ )]
:attr:`kB.J`    Boltzmann constant [J/K]
:attr:`kB.eV`   Boltzmann constant [eV/K]
=============== ============================================================

'''

class Quantity(object):
    '''Physical quantity.

       Use as ``quantity.unit`` to get its value
    '''
    pass

qe = 1.60217733e-19

c = 299792458.

mu0 = 1.25663706144e-06

eps0 = 8.854187817e-6

h = Quantity()
h.J = 6.62606957e-34
h.eV = 4.135667516e-15

SB = 5.670373e-8

kB = Quantity()
kB.J = 1.3806503e-23
kB.eV = 8.6173423e-5

del Quantity
