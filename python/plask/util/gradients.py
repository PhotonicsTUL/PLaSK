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

"""Gradients simplification.

This module contain functions to simplify gradient layers for optical calculations.
Each such layer is replaced with two uniform layers providing the same transfer matrix.
Using such simplified gradients for optical computations strongly improves the accuracy
and speed of optical computations.

For details see: M. Wasiak, M. Dobrski...
"""

from .. import tensor as _tensor, print_log as _log
from .. import geometry as _geometry
from ..material import Material as _Material


class _SimplifiedMaterial(_Material):

    def __init__(self, base, Nr, lam, T, dNdl, dNdT):
        super().__init__(base)
        self._nr = Nr.real
        fa = - 125663706.14366701 / lam
        self._absp = fa * Nr.imag
        self._lam = lam
        self._T = T
        self._dndl = dNdl.real
        self._dndT = dNdT.real
        self._dadT = fa * dNdT.imag

    def __str__(self):
        return str(self.base)

    def nr(self, lam, T=300., n=0.):
        return self._nr + self._dndl * (lam - self._lam) + self._dndT * (T - self._T)

    def absp(self, lam, T=300.):
        return self._absp + self._dadT * (T - self._T)

    def Nr(self, lam, T=300., n=0.):
        return self.nr(lam, T, n) - 7.95774715459e-09j * self.absp(lam, T) * lam

    def Eps(self, lam, T=300., n=0.):
        eps = self.Nr(lam, T, n)**2
        return _tensor(eps, eps, eps)


def simplify(item, lam, T=300., linear='nr', dT=100.):
    """
    Return stack of two layers providing the same optical parameters as the gradient
    layer.

    Args:
        item (~plask.geometry.Rectangle or ~plask.geometry.Cuboid): object to simplify
        lam (float): reference wavelength
        T (float): temperature for the refractive indices
        linear ('nr' or 'eps'): which parameter should be linear
        dT (float): temperature step for estimation of temperature dependence
                    of refractive index

    Returns:
        ~plask.geometry.GeometryObject: an object replacing the original leaf
    """
    if isinstance(item, _geometry.Rectangle):
        Block = _geometry.Rectangle
        Stack = _geometry.SingleStack2D
    elif isinstance(item, _geometry.Cuboid):
        Block = _geometry.Cuboid
        Stack = _geometry.SingleStack3D
    else:
        raise TypeError("'item' must be Rectangle or Cuboid")

    try:
        m0, m1 = item.material
    except TypeError:   # the object is uniform
        return item

    material = item.representative_material

    from ._gradients import simplify_gradient_nr

    dl = 1.

    data0 = simplify_gradient_nr(item.height, m0.Nr(lam, T), m1.Nr(lam, T), lam, linear)
    dataT = simplify_gradient_nr(item.height, m0.Nr(lam, T+dT), m1.Nr(lam, T+dT), lam, linear, data0[0][0])
    datal = simplify_gradient_nr(item.height, m0.Nr(lam+dl, T), m1.Nr(lam+dl, T), lam+dl, linear, data0[0][0])

    _log('debug', '{0}->{1} ({2:.6f}nm, {3}K): {4[0]:.3f}[{4[1]:.4f}um] / {4[0]:.3f}[{4[1]:.4f}um]'
         .format(m0, m1, lam, T, *data0))
    _log('debug', '{0}->{1} ({2:.6f}nm, {3}K): {4[0]:.3f}[{4[1]:.4f}um] / {4[0]:.3f}[{4[1]:.4f}um] (dT)'
         .format(m0, m1, lam, T+dT, *dataT))
    _log('debug', '{0}->{1} ({2:.6f}nm, {3}K): {4[0]:.3f}[{4[1]:.4f}um] / {4[0]:.3f}[{4[1]:.4f}um] (dlam)'
         .format(m0, m1, lam+dl, T, *datal))

    stack = Stack()
    dims = list(item.dims)
    for (Nr, d), (NrT, _), (Nrl, _) in zip(data0, dataT, datal):
        dims[-1] = d
        dNdT = (NrT - Nr) / dT
        nNdl = (Nrl - Nr) / dl
        stack.append(Block(dims, _SimplifiedMaterial(material, Nr, lam, T, nNdl, dNdT)))
    stack.roles = item.roles | {'__gradient'}

    return stack


def simplify_all(geometry, lam, T=300., linear='nr', dT=100.):
    """
    Replace all rectangular blocks with gradients with two-layer simplification
    in geometry tree.

    Args:
        geometry (~plask.geometry.GeometryObject): object to simplify
        lam (float): reference wavelength
        T (float): temperature for the refractive indices
        linear ('nr' or 'eps'): which parameter should be linear
        dT (float): temperature step for estimation of temperature dependence
                    of refractive index

    Returns:
        ~plask.geometry.GeometryObject: geometry with simplified gradients
    """

    def filter(item):
        if isinstance(item, (_geometry.Rectangle, _geometry.Cuboid)):
            new_item = simplify(item, lam, T, linear, dT)
            if new_item is not item:
                return new_item

    return geometry.modify_objects(filter)
