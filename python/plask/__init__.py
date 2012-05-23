# -*- coding: utf-8 -*-
from .plaskcore import *

from numpy import *

## ## plask.material ## ##

materialdb = material.database = material.MaterialsDB.getDefault()

def updateFactories():
    '''For each material in default database make factory in plask.material'''
    def factory(name):
        return lambda **kwargs: materialdb.get(name, **kwargs)
    for mat in material.database:
        name = mat.split(":")[0]
        if name not in material.__dict__:
            material.__dict__[name] = factory(name)
material.updateFactories = updateFactories
del updateFactories

def importLibrary(name):
    from ctypes import cdll
    for lib in ["lib"+name+".so", "lib"+name, name+".so", name]:
        try:
            cdll.LoadLibrary(lib)
        except OSError:
            pass
        else:
            material.updateFactories()
            return
    raise OSError("Cannot import library '%s'" % name)
material.importLibrary = importLibrary
del importLibrary

material.importLibrary("plask_materialsdefault")

def register_material(cls=None, name=None, complex=False, DB=None):
    '''Register a custom Python material'''

    # A trick allowing passing arguments to decorator
    if cls is None:
        return lambda M: register_material(M, name=name, DB=DB)
    elif not issubclass(cls, material.Material):
        raise TypeError("Wrong decorated class (must be a subclass of plask.material.Material")

    if 'name' in cls.__dict__:
        if name is not None:
            raise ValueError("Name specified both in decorator parameter and class body")
    elif name is not None:
        cls.name = name
    else:
        cls.name = cls.__name__

    if DB is None:
        DB = material.database

    if complex:
        material._register_material_complex(cls.name, cls, DB)
    else:
        material._register_material_simple(cls.name, cls, DB)

    return cls

material.register_material = register_material
del register_material

material.simple = lambda mat, **kwargs: material.register_material(mat, complex=False, **kwargs)
material.complex = lambda mat, **kwargs: material.register_material(mat, complex=True, **kwargs)


## ## plask.geometry ## ##



## ##  ## ##
