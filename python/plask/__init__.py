from modplask import *

## ## plask.material ## ##

def registerMaterial(Material):
    '''Function to register a new material'''
    if not 'name' in Material.__dict__:
        Material.name = Material.__name__
    material._registerMaterial(Material.name, Material) # register to C++
    return Material

material.new = material.registerMaterial = registerMaterial
#del registerMaterial

## ## plask.geometry ## ##



## ##  ## ##
