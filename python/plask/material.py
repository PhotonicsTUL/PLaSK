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

"""
Materials and material database.

Many semiconductor materials used in photonics are defined here. We have made
a significant effort to ensure their physical properties to be the most precise
as the current state of the art. However, you can derive an abstract class
:class:`plask.Material` to create your own materials.
"""

import os as _os

try:
    from . import _material
except ImportError:
    from ._plask import _material

from ._material import *


db = _material.getdb()

get = db.get
with_params = db.material_with_params


def update_factories():
    """For each material in default database make factory in ``plask.material``."""
    def factory(name):
        def func(**kwargs):
            if 'dopant' in kwargs:
                kwargs = kwargs.copy()
                dopant = kwargs.pop('dopant')
                return db.get(name+':'+dopant, **kwargs)
            else:
                return db.get(name, **kwargs)
        func.__doc__ = u"Create material {}.\n\n:rtype: Material".format(name)
        return func
    for mat in db:
        if mat == 'air': continue
        name = mat.split(":")[0]
        if name not in globals():
            globals()[name.replace('-', '_')] = factory(name)


air = get("air")

Air = lambda: air

if 'PLASK_DEFAULT_MATERIALS' not in _os.environ:
    _material.load_all_libraries()
else:
    for _ml in _os.environ['PLASK_DEFAULT_MATERIALS'].split(';' if _os.name == 'nt' else ':'):
        if _ml: _material.load_library(_ml)
update_factories()


def load_library(lib):
    """
    Load material library from file to database.

    Mind that this function will load each library only once (even if
    the database was cleared).

    Args:
        lib (str): Library to load without the extension (.so or .dll).
    """
    _material.load_library(lib)
    update_factories()

def load_all_libraries(dir):
    """
    Load all materials from specified directory to database.

    This method can be used to extend the database with custom materials provided
    in binary libraries.

    Mind that this function will load each library only once (even if
    the database was cleared).

    Args:
        dir (str): Directory name to load materials from.
    """
    _material.load_all_libraries(dir)
    update_factories()


class simple:
    """
    Decorator for custom simple material class.

    Args:
        base (str or material.Material): Base class specification.
            It may be either a material object or a string with either
            complete or incomplete specification of the material.
            In either case you must initialize the base in the
            constructor of your material.
    """
    def __init__(self, base=None):
        if isinstance(base, type):
            raise TypeError("@material.simple argument is a class (you probably forgot parenthes)")
        self.base = base
    def __call__(self, cls):
        if 'name' not in cls.__dict__: cls.name = cls.__name__
        _material._register_material_simple(cls.name, cls, self.base)
        return cls


class alloy:
    """
    Decorator for custom alloy material class.

    Args:
        base (str or material.Material): Base class specification.
            It may be either a material object or a string with either
            complete or incomplete specification of the material.
            In either case you must initialize the base in the
            constructor of your material.
    """
    def __init__(self, base=None):
        if isinstance(base, type):
            raise TypeError("@material.alloy argument is a class (you probably forgot brackets)")
        self.base = base
    def __call__(self, cls):
        if 'name' not in cls.__dict__: cls.name = cls.__name__
        _material._register_material_alloy(cls.name, cls, self.base)
        return cls


const = staticmethod
