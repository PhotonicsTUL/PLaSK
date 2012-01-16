# -*- coding: utf-8 -*-
from modplask import *


## ## plask.material ## ##

material.database = material.MaterialsDB()
for m in material.database:
    material.__dict__[m] = lambda name, *args, **kwargs: material.database.factory(name, args, kwargs)

def registerMaterial(Material, DB=None):
    '''Function to register a new material'''

    # A trick allowing passing arguments to decorator
    if isinstance(Material, material.MaterialsDB):
        if DB is not None:
            raise TypeError("Wrong arguments in decorator")
        else:
            return lambda M: registerMaterial(M, Material)
    else:
        if not issubclass(Material, material.Material):
            raise TypeError("Material class must be a subclass of plask.material.Material")

    if 'name' not in Material.__dict__:
        Material.name = lambda self: Material.__name__

    if DB is None:
        DB = material.database
    material.registerMaterial(Material.name(Material()), Material, DB) # register to C++

    return Material

material.new = registerMaterial
#del registerMaterial



## ## plask.geometry ## ##



## ##  ## ##
