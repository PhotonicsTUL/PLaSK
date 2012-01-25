# -*- coding: utf-8 -*-
from plaskcore import *


## ## plask.material ## ##

material.database = material.MaterialsDB()

# Create factories for default materials
for mat in material.database:
    material.__dict__[mat] = lambda **kwargs: material.database.get(mat.split(":")[0], **kwargs)

def registerMaterial(Material=None, name=None, DB=None):
    '''Function to register a new material'''

    # A trick allowing passing arguments to decorator
    if Material is None:
        return lambda M: registerMaterial(M, name=name, DB=DB)
    elif not issubclass(Material, material.Material):
        raise TypeError("Wrong decorated class (must be a subclass of plask.material.Material")

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
