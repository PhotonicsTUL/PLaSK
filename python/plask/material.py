# -*- coding: utf-8 -*-
"""
Materials and material database.

Many semiconductor materials used in photonics are defined here. We have made
a significant effort to ensure their physical properties to be the most precise
as the current state of the art. However, you can derive an abstract class
:class:`plask.Material` to create your own materials.
"""

try:
    from . import _material
except ImportError:
    from ._plask import _material

from ._material import *


db = _material.getdb()

get = db.get


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

_material.load_all_libraries()
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

def load_all_libraries(lib):
    """
    Load all materials from specified directory to database.

    This method can be used to extend the database with custom materials provided
    in binary libraries.

    Mind that this function will load each library only once (even if
    the database was cleared).

    Args:
        dir (str): Directory name to load materials from.
    """
    _material.load_all_libraries(lib)
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
            raise TypeError("@material.alloy argument is a class (you probably forgot parenthes)")
        self.base = base
    def __call__(self, cls):
        if 'name' not in cls.__dict__: cls.name = cls.__name__
        _material._register_material_alloy(cls.name, cls, self.base)
        return cls

class complex(alloy):
    def __init__(self, base=None):
        from . import print_log
        print_log('warning', "Decorator @material.complex is obsolete, use @material.alloy instead")
        super(complex, self).__init__(base)


const = staticmethod
