# -*- coding: utf-8 -*-
from plaskcore import *

from numpy import *

## ## plask.materials ## ##

materialsdb = materials.database = materials.MaterialsDB.getDefault()

def updateFactories():
    '''For each material in default database make factory in plask.materials'''
    def factory(name):
        return lambda **kwargs: materialsdb.get(name, **kwargs)
    for mat in materials.database:
        name = mat.split(":")[0]
        if name not in materials.__dict__:
            materials.__dict__[name] = factory(name)
materials.updateFactories = updateFactories
del updateFactories
materials.updateFactories()

def register_material(cls=None, name=None, complex=False, DB=None):
    '''Register a custom Python material'''

    # A trick allowing passing arguments to decorator
    if cls is None:
        return lambda M: register_material(M, name=name, DB=DB)
    elif not issubclass(cls, materials.Material):
        raise TypeError("Wrong decorated class (must be a subclass of plask.materials.Material")

    if 'name' in cls.__dict__:
        if name is not None:
            raise ValueError("Name specified both in decorator parameter and class body")
    elif name is not None:
        cls.name = name
    else:
        cls.name = cls.__name__

    if DB is None:
        DB = materials.database

    if complex:
        materials._register_material_complex(cls.name, cls, DB)
    else:
        materials._register_material_simple(cls.name, cls, DB)

    return cls

materials.register_material = register_material
del register_material

materials.simple = lambda mat, **kwargs: materials.register_material(mat, complex=False, **kwargs)
materials.complex = lambda mat, **kwargs: materials.register_material(mat, complex=True, **kwargs)


## ## plask.geometry ## ##



## ##  ## ##
