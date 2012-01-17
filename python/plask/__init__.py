# -*- coding: utf-8 -*-
from modplask import *


## ## plask.material ## ##

material.database = material.MaterialsDB()
for m in material.database:
    material.__dict__[m] = lambda name, *args, **kwargs: material.database.factory(name, args, kwargs)

def registerMaterial(Material=None, name=None, DB=None):
    '''Function to register a new material'''

    # A trick allowing passing arguments to decorator
    if Material is None:
        return lambda M: registerMaterial(M, name=name, DB=DB)
    elif not issubclass(Material, material.Material):
        raise TypeError("Wrong argument to decorator (must be a subclass of plask.material.Material")

    if 'name' in Material.__dict__:
        if name is not None:
            raise ValueError("Name specified both in decorator parameter and class body")
    elif name is not None:
        Material.name = name
    else:
        Material.name = Material.__name__

    if DB is None:
        DB = material.database
    material.registerMaterial(Material.name, Material, DB) # register to C++

    return Material

material.new = registerMaterial



## ## plask.geometry ## ##



## ##  ## ##)
