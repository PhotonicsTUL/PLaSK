from modplask import *

## ## plask.material ## ##

material.database = material.MaterialsDB()

def registerMaterial(Material, DB=None):
    '''Function to register a new material'''
    if 'name' not in Material.__dict__:
        Material.name = lambda self: Material.__name__
    if DB is None:
        DB = material.database
    material._registerMaterial(Material.name(Material()), Material, DB) # register to C++
    return Material

material.new = material.registerMaterial = registerMaterial
#del registerMaterial

## ## plask.geometry ## ##



## ##  ## ##
