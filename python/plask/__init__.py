# -*- coding: utf-8 -*-
from plaskcore import *

from numpy import *

## ## plask.material ## ##

materials = material.database

# Create factories for default materials
for mat in material.database:
    material.__dict__[mat] = lambda **kwargs: material.database.get(mat.split(":")[0], **kwargs)

def register_material(Material=None, name=None, complex=False, DB=None):
    '''Function to register a new material'''

    # A trick allowing passing arguments to decorator
    if Material is None:
        return lambda M: register_material(M, name=name, DB=DB)
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

    if complex:
        material._register_material_complex(Material.name, Material, DB)
    else:
        material._register_material_simple(Material.name, Material, DB)

    return Material

material.register_material = register_material
del register_material

material.simple = lambda mat, **kwargs: material.register_material(mat, complex=False, **kwargs)
material.complex = lambda mat, **kwargs: material.register_material(mat, complex=True, **kwargs)


## ## plask.geometry ## ##



## ##  ## ##
