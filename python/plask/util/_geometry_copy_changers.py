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

from .. import Manager, geometry

@Manager._geometry_changer('simplify-gradients')
class GradientChanger:

    def __init__(self, xpl, manager):
        self.lam = xpl['lam']
        self.T = xpl.get('temp', 300.)
        self.linear = xpl.get('linear', 'nr')
        if self.linear != 'eps' and self.linear.lower() != 'nr':
            raise ValueError("'linear' argument must be either 'eps' or 'nr'")
        self.dT = xpl.get('dtemp', 100.)
        self.only_role = xpl.get('only-role')

    def __call__(self, item):
        from .gradients import simplify
        if isinstance(item, (geometry.Rectangle, geometry.Cuboid)) and (self.only_role is None or self.only_role in item.roles):
            new_item = simplify(item, self.lam, self.T, self.linear, self.dT)
            if new_item is not item:
                return new_item
