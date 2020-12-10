#coding: UTF-8
"""
Basic physical constants and utility functions.

Constants
=========

=============== ============================================================
:attr:`qe`      Elementary charge [C]
:attr:`me`      Electron mass [kg]
:attr:`c`       Speed of light [m/s]
:attr:`mu0`     Vacuum permeability [V·s/(A·m)]
:attr:`eps0`    Vacuum permittivity [pF/µm]
:attr:`eta0`    Free space impedance [Ω]
:attr:`Z0`      Free space impedance [Ω]
:attr:`h.J`     Pnav-sidenavlanck constant [J·s]
:attr:`h.eV`    Planck constant [eV·s]
:attr:`SB`      Stefan-Boltzmann constant [W/(m\ :sup:`2`\ ·K\ :sup:`4`\ )]
:attr:`kB.J`    Boltzmann constant [J/K]
:attr:`kB.eV`   Boltzmann constant [eV/K]
=============== ============================================================
"""

from plask import material as _material


qe = 1.60217733e-19

me = 9.10938291e-31

c = 299792458.

mu0 = µ0 = 1.25663706144e-06

eps0 = ε0 = 8.854187817e-6

Z0 = eta0 = η0 = 376.73031346177066

class h:
    J = 6.62606957e-34
    eV = 4.135667516e-15
h = h()

SB = 5.670373e-8


class kB:
    J = 1.3806503e-23
    eV = 8.6173423e-5
kB = kB()


def wl(mat, lam, T=300.):
    """
    Compute real length of optical wavelength in specified material.

    This is utility function that computes the physical length of a single
    wavelength in specified material. Its main purpose is easier design of
    DBR stacks.

    If you are using it with custom materials, make sure that it does provide
    :meth:`~plask.material.Material.nr` method.

    Args:
        mat (material.Material or str): Material to compute physical wavelength in.
        lam (float): Free-space wavelength to scale for material `mat` [nm].
        T (float): Temperature at which material refractive index is retrieved.

    Returns:
        float: Real length of the one wavelength in the material [µm].
    """
    return 1e-3 * lam / _material.get(mat).Nr(lam, T).real


def eV2nm(eV):
    """
    Compute wavelength for specified photon energy.

    Args:
        eV (float): Photon energy [eV].

    Returns:
        float: Wavelength [nm].
    """
    return 1239.84193009 / eV


def nm2eV(nm):
    """
    Compute photon energy for specified wavelength.

    Args:
        nm (float): Wavelength [eV].

    Returns:
        float: Photon energy [nm].
    """
    return 1239.84193009 / nm
